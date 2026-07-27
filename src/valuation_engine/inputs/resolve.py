"""Merge asset + modality + territory (+ overrides + clinical LoA) into a flat
parameter set the deterministic model can consume.

The output, :class:`ResolvedInputs`, holds:

* ``sourced`` – an ordered mapping ``name -> SourcedValue`` for every scalar
  input. This drives both the base case (``base_params``) and Monte Carlo
  (``sample_params``), and is the assumptions ledger for reporting.
* structural config that isn't a single number: dosing shape, epi basis,
  horizon, the list of competitors, and the deal terms.

Override precedence for asset×territory facts: ``AssetSpec.territory_overrides``
> ``TerritoryPack`` default. A direct ``net_price_usd`` override is folded in by
setting ``list_price`` to it and ``reference_price_factor`` to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from valuation_engine.inputs.schemas import (
    AssetSpec,
    CompetitorEntry,
    DealTerms,
    ModalityPack,
    SourcedValue,
    TerritoryAssetInputs,
    TerritoryPack,
)

# Default probability that, conditional on clinical success, the asset clears
# territory-level regulatory review. High because in-territory approval mostly
# follows a successful global filing; overridable per asset×territory.
DEFAULT_P_TERRITORY_APPROVAL = 0.9


@dataclass
class ResolvedInputs:
    """Flat, model-ready inputs for one (asset, territory) valuation."""

    asset_id: str
    territory_id: str
    dosing: str
    epi_basis: str
    horizon_years: int
    deal: DealTerms
    competitors: list[str]  # names, in the order their params are keyed
    sourced: dict[str, SourcedValue] = field(default_factory=dict)

    # -- parameter access --------------------------------------------------- #
    def base_params(self) -> dict[str, float]:
        return {k: v.base() for k, v in self.sourced.items()}

    def sample_params(
        self, rng: np.random.Generator, n: int
    ) -> dict[str, np.ndarray]:
        return {k: v.distribution().sample(rng, n) for k, v in self.sourced.items()}

    def ledger(self) -> list[tuple[str, SourcedValue]]:
        return list(self.sourced.items())


def _pick(
    override: Optional[SourcedValue], default: SourcedValue
) -> SourcedValue:
    return override if override is not None else default


def resolve(
    asset: AssetSpec,
    modality: ModalityPack,
    territory: TerritoryPack,
    clinical_loa: SourcedValue,
    deal: Optional[DealTerms] = None,
) -> ResolvedInputs:
    """Produce a :class:`ResolvedInputs` for one asset×territory.

    ``clinical_loa`` is the probability-of-launch supplied by the
    :class:`~valuation_engine.clinical.provider.ClinicalModelProvider` (or the
    asset's own override if present). It is applied as a risk weight; the
    engine does not model clinical trials itself.
    """
    if modality.id != asset.modality_id:
        raise ValueError(
            f"modality pack {modality.id!r} does not match asset modality {asset.modality_id!r}"
        )

    ov: TerritoryAssetInputs = asset.territory_overrides.get(
        territory.id, TerritoryAssetInputs()
    )
    epi = asset.epidemiology
    s: dict[str, SourcedValue] = {}

    # -- epidemiology ------------------------------------------------------- #
    s["population"] = territory.population
    s["epi_rate_per_100k"] = epi.rate_per_100k
    s["diagnosis_rate"] = epi.diagnosis_rate
    s["treatment_rate"] = epi.treatment_rate
    s["eligible_fraction"] = epi.eligible_fraction

    # -- pricing ------------------------------------------------------------ #
    if ov.net_price_usd is not None:
        # Direct net-price override: fold into list_price with factor 1.
        s["list_price"] = ov.net_price_usd
        s["reference_price_factor"] = SourcedValue(value=1.0)
    else:
        s["list_price"] = asset.global_list_price_usd
        s["reference_price_factor"] = _pick(
            ov.reference_price_factor, territory.reference_price_factor
        )
    s["public_gtn_discount"] = territory.public_gtn_discount
    s["private_gtn_discount"] = territory.private_gtn_discount
    s["annual_price_erosion"] = territory.annual_price_erosion
    s["loe_erosion"] = asset.loe_erosion
    s["exclusivity_years"] = asset.exclusivity_years

    # -- access / basket ---------------------------------------------------- #
    s["p_territory_approval"] = _pick(
        ov.p_territory_approval,
        SourcedValue(value=DEFAULT_P_TERRITORY_APPROVAL, kind="bernoulli"),
    )
    p_reimb = _pick(ov.p_reimbursement, territory.p_reimbursement_default)
    # Ensure reimbursement is treated as a binary basket event in MC.
    if p_reimb.kind == "point":
        p_reimb = p_reimb.model_copy(update={"kind": "bernoulli"})
    s["p_reimbursement"] = p_reimb
    s["private_channel_share"] = territory.private_channel_share

    # -- timing ------------------------------------------------------------- #
    s["regulatory_review_years"] = _pick(
        ov.regulatory_review_years, territory.regulatory_review_years
    )
    s["reimbursement_lag_years"] = _pick(
        ov.reimbursement_lag_years, territory.reimbursement_lag_years
    )
    s["launch_delay_years"] = (
        ov.launch_delay_years
        if ov.launch_delay_years is not None
        else SourcedValue(value=0.0)
    )

    # -- competition / uptake ---------------------------------------------- #
    s["peak_share"] = _pick(ov.peak_share, asset.peak_share)
    s["uptake_years_to_peak"] = asset.uptake_years_to_peak
    s["differentiation_score"] = asset.differentiation_score

    competitors: list[CompetitorEntry] = ov.competitors
    comp_names: list[str] = []
    for i, c in enumerate(competitors):
        comp_names.append(c.name)
        launch = (
            SourcedValue(value=0.0) if c.available_now else c.launch_year_from_now
        )
        s[f"comp{i}_launch_year"] = launch
        s[f"comp{i}_share_capture"] = c.share_capture

    # -- costs -------------------------------------------------------------- #
    s["cogs_pct"] = modality.default_cogs_pct
    s["sga_pct"] = territory.sga_pct
    s["distribution_pct"] = territory.distribution_pct
    s["treatment_duration_years"] = (
        asset.treatment_duration_years_override
        if asset.treatment_duration_years_override is not None
        else modality.default_treatment_duration_years
    )
    s["one_time_annual_replenishment"] = modality.one_time_annual_replenishment
    s["filing_cost_usd"] = territory.filing_cost_usd
    s["market_access_spend_usd"] = _pick(
        ov.market_access_spend_usd, territory.market_access_spend_usd
    )
    s["bridging_trial_cost_usd"] = territory.bridging_trial_cost_usd

    # -- risk & discounting ------------------------------------------------- #
    loa = asset.clinical_loa_override if asset.clinical_loa_override is not None else clinical_loa
    s["clinical_loa"] = loa
    s["base_discount_rate"] = territory.base_discount_rate
    s["country_risk_premium"] = territory.country_risk_premium
    s["fx_to_usd"] = territory.fx_to_usd

    # -- deal --------------------------------------------------------------- #
    deal = deal or DealTerms()
    s["upfront_usd"] = deal.upfront_usd
    for i, m in enumerate(deal.milestones):
        s[f"milestone{i}_amount"] = m.amount_usd

    return ResolvedInputs(
        asset_id=asset.id,
        territory_id=territory.id,
        dosing=modality.dosing,
        epi_basis=epi.basis,
        horizon_years=territory.horizon_years,
        deal=deal,
        competitors=comp_names,
        sourced=s,
    )
