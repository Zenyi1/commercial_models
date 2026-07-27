"""Risk-adjusted NPV assembler — the heart of the engine.

``run_model`` takes a :class:`~valuation_engine.inputs.resolve.ResolvedInputs`
and a flat ``params`` mapping (scalar floats) and produces the full annual
forecast and the resulting values. It is deliberately written to run on a single
scalar parameter draw so that:

* the **base case** passes ``resolved.base_params()`` (probabilities act as
  expected-value weights), and
* **Monte Carlo** calls it once per sampled draw (binary gates realised 0/1),

through one identical code path — no separate "deterministic" and "stochastic"
models that could drift apart.

Probability weighting is applied consistently: the launch gate
``clinical_loa x p_territory_approval`` scales the entire revenue stream, and
``p_reimbursement`` gates the public channel. In the base case these are
probabilities; in Monte Carlo they are 0/1, which is what produces the realistic
(often bimodal) value distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from valuation_engine.core.timeline import npv
from valuation_engine.inputs.resolve import ResolvedInputs
from valuation_engine.model import (
    access,
    competition,
    costs,
    epidemiology,
    pricing,
    timing,
    uptake,
)
from valuation_engine.model.deal import DealValuation, value_deal


@dataclass
class ModelResult:
    """Everything a single model run produces, for reporting and aggregation."""

    rate: float
    market_entry_year: float
    public_start_year: float
    launch_weight: float  # clinical_loa * p_territory_approval used in this run
    years: np.ndarray
    treated_patients: np.ndarray
    net_price_public: np.ndarray
    net_price_private: np.ndarray
    net_sales: np.ndarray
    operating_cashflow: np.ndarray
    pie_rnpv: float
    cost_to_enter_pv: float
    net_value: float  # pie_rnpv - cost_to_enter_pv
    deal: DealValuation

    @property
    def peak_net_sales(self) -> float:
        return float(np.max(self.net_sales)) if self.net_sales.size else 0.0


def _competitor_arrays(
    resolved: ResolvedInputs, params: Mapping[str, float]
) -> tuple[list[float], list[float]]:
    launches, shares = [], []
    for i in range(len(resolved.competitors)):
        launches.append(float(params[f"comp{i}_launch_year"]))
        shares.append(float(params[f"comp{i}_share_capture"]))
    return launches, shares


def run_model(
    resolved: ResolvedInputs,
    params: Mapping[str, float],
    convention: str = "mid",
) -> ModelResult:
    horizon = resolved.horizon_years
    years = np.arange(horizon)

    # -- timing, discounting, launch gate ---------------------------------- #
    entry = timing.market_entry_year(
        params["regulatory_review_years"], params["launch_delay_years"]
    )
    pub_start = timing.public_start_year(entry, params["reimbursement_lag_years"])
    rate = timing.discount_rate(
        params["base_discount_rate"], params["country_risk_premium"]
    )
    launch_weight = timing.commercialization_weight(
        params["clinical_loa"], params["p_territory_approval"]
    )

    # -- epidemiology ------------------------------------------------------- #
    base_elig = epidemiology.base_eligible(
        params["population"],
        params["epi_rate_per_100k"],
        params["diagnosis_rate"],
        params["treatment_rate"],
        params["eligible_fraction"],
    )

    # -- price trajectories ------------------------------------------------- #
    gp = pricing.gross_price(params["list_price"], params["reference_price_factor"])
    net_public = pricing.net_price_trajectory(
        years, entry, gp, params["public_gtn_discount"],
        params["annual_price_erosion"], params["exclusivity_years"], params["loe_erosion"],
    )
    net_private = pricing.net_price_trajectory(
        years, entry, gp, params["private_gtn_discount"],
        params["annual_price_erosion"], params["exclusivity_years"], params["loe_erosion"],
    )

    # -- share (competition-adjusted uptake) ------------------------------- #
    peak = competition.adjusted_peak_share(
        params["peak_share"], params["differentiation_score"]
    )
    u = uptake.uptake_fraction(years, entry, params["uptake_years_to_peak"])
    comp_launches, comp_shares = _competitor_arrays(resolved, params)
    retention = competition.competition_retention(years, comp_launches, comp_shares)
    share = peak * u * retention  # recurring: share of pool; one_time: annual capture rate

    # -- channel reach ------------------------------------------------------ #
    priv_frac, pub_frac = access.channel_fractions(
        years, entry, pub_start,
        params["private_channel_share"], params["p_reimbursement"],
    )
    # Self-pay reach is gated by affordability: the private channel shrinks as
    # the (per-year) net self-pay price climbs relative to local income. This is
    # why an unreimbursed cheap drug retains most of its value out of pocket
    # while an unreimbursed six-figure therapy does not.
    afford = access.affordability_factor(
        net_private, params["gdp_per_capita"],
        params["affordability_multiple"], params["affordability_steepness"],
    )
    priv_frac = priv_frac * afford

    # -- patients & revenue ------------------------------------------------- #
    if resolved.dosing == "recurring":
        pool = epidemiology.recurring_on_therapy_stock(
            base_elig, resolved.epi_basis, params["treatment_duration_years"]
        )
        treated_private = pool * priv_frac * share
        treated_public = pool * pub_frac * share
    else:  # one_time: draw down the prevalent bolus, with incident replenishment
        treated_private = np.zeros(horizon)
        treated_public = np.zeros(horizon)
        remaining = base_elig
        replenish = params["one_time_annual_replenishment"] * base_elig
        for y in range(horizon):
            cap_private = remaining * priv_frac[y] * share[y]
            cap_public = remaining * pub_frac[y] * share[y]
            treated_private[y] = cap_private
            treated_public[y] = cap_public
            remaining = max(remaining - (cap_private + cap_public) + replenish, 0.0)

    treated = treated_private + treated_public
    revenue = treated_private * net_private + treated_public * net_public

    # Launch gate scales the whole stream (expected value in base; 0/1 in MC).
    net_sales = revenue * launch_weight

    # -- operating cash flow (the intrinsic "pie") ------------------------- #
    margin = costs.operating_margin(
        params["cogs_pct"], params["sga_pct"], params["distribution_pct"]
    )
    operating_cashflow = net_sales * margin
    pie_rnpv = npv(rate, operating_cashflow, convention)

    # -- cost to enter (kept separate from the pie) ------------------------ #
    cost_to_enter = costs.cost_to_enter_pv(
        params["filing_cost_usd"], params["bridging_trial_cost_usd"],
        params["market_access_spend_usd"], rate, entry, convention,
    )
    net_value = pie_rnpv - cost_to_enter

    # -- deal split --------------------------------------------------------- #
    deal_val = value_deal(
        pie_rnpv=pie_rnpv,
        net_sales=net_sales,
        years=years,
        rate=rate,
        params=params,
        milestones=resolved.deal.milestones,
        royalty_tiers=resolved.deal.royalty_tiers,
        market_entry=entry,
        public_start=pub_start,
        commercialization_weight=launch_weight,
        p_reimbursement=params["p_reimbursement"],
        convention=convention,
    )

    return ModelResult(
        rate=rate,
        market_entry_year=entry,
        public_start_year=pub_start,
        launch_weight=launch_weight,
        years=years,
        treated_patients=treated,
        net_price_public=net_public,
        net_price_private=net_private,
        net_sales=net_sales,
        operating_cashflow=operating_cashflow,
        pie_rnpv=pie_rnpv,
        cost_to_enter_pv=cost_to_enter,
        net_value=net_value,
        deal=deal_val,
    )


def evaluate_base(resolved: ResolvedInputs, convention: str = "mid") -> ModelResult:
    """Convenience: run the deterministic base case."""
    return run_model(resolved, resolved.base_params(), convention)
