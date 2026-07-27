"""Known-answer and invariant tests for the model modules and the assembler."""

import numpy as np
import pytest

from valuation_engine.inputs.schemas import (
    AssetSpec,
    DealTerms,
    EpidemiologyBlock,
    MacroBlock,
    Milestone,
    ModalityPack,
    RoyaltyTier,
    SourcedValue,
    TerritoryPack,
)
from valuation_engine.inputs.resolve import resolve
from valuation_engine.model import competition, epidemiology, pricing, uptake
from valuation_engine.model.access import affordability_factor, channel_fractions
from valuation_engine.model.deal import royalty_by_year
from valuation_engine.rnpv import evaluate_base, run_model


# --------------------------------------------------------------------------- #
# Model-module unit tests
# --------------------------------------------------------------------------- #
def test_base_eligible_known_answer():
    # 100M pop, 500/100k prevalence, 60% dx, 50% tx, 20% eligible
    v = epidemiology.base_eligible(100_000_000, 500, 0.6, 0.5, 0.2)
    assert v == pytest.approx(100_000_000 * 0.005 * 0.6 * 0.5 * 0.2)  # 30,000


def test_recurring_stock_incidence_vs_prevalence():
    assert epidemiology.recurring_on_therapy_stock(1000, "prevalence", 5) == 1000
    assert epidemiology.recurring_on_therapy_stock(1000, "incidence", 5) == 5000


def test_price_zero_before_entry_and_loe_step():
    years = np.arange(10)
    p = pricing.net_price_trajectory(
        years, market_entry=2.0, gross_price_value=100.0, gtn_discount=0.2,
        annual_erosion=0.10, exclusivity_years=3.0, loe_erosion=0.2,
    )
    assert p[0] == 0.0 and p[1] == 0.0  # mid-year 0.5, 1.5 < 2
    # year 2 (mid 2.5): base_net 80 * 0.9^0.5
    assert p[2] == pytest.approx(80 * 0.9**0.5, rel=1e-6)
    # year 5 (mid 5.5) is past LoE (entry 2 + 3): apply loe_erosion
    assert p[5] == pytest.approx(80 * 0.9**3.5 * 0.2, rel=1e-6)


def test_uptake_monotone_between_zero_and_one():
    years = np.arange(12)
    u = uptake.uptake_fraction(years, market_entry=1.0, years_to_peak=5.0)
    active = u[1:]
    assert np.all(np.diff(active) >= -1e-12)  # non-decreasing after entry
    assert u.min() >= 0.0 and u.max() <= 1.0 + 1e-9


def test_competition_retention_only_bites_after_launch():
    years = np.arange(8)
    ret = competition.competition_retention(years, [3.0], [0.4])
    assert ret[0] == pytest.approx(1.0)  # before competitor launch
    assert ret[5] == pytest.approx(0.6)  # after: 1 - 0.4


def test_adjusted_peak_share_capped():
    assert competition.adjusted_peak_share(0.5, 2.0) == pytest.approx(0.9)  # capped
    assert competition.adjusted_peak_share(0.3, 1.2) == pytest.approx(0.36)


def test_channel_fractions_gating():
    years = np.arange(6)
    priv, pub = channel_fractions(
        years, market_entry=1.0, public_start=3.0,
        private_channel_share=0.2, p_reimbursement=0.5,
    )
    assert priv[0] == 0.0 and priv[2] == pytest.approx(0.2)  # private from entry
    assert pub[2] == 0.0  # before public start
    assert pub[4] == pytest.approx((1 - 0.2) * 0.5)  # public gated by reimbursement


def test_affordability_factor_price_wall():
    gdp = 14000.0
    # Cheap drug -> near-full self-pay reach; at the reference price -> ~half;
    # far above -> collapses toward zero.
    cheap = affordability_factor(300.0, gdp, 2.0, 2.5)
    at_ref = affordability_factor(2.0 * gdp, gdp, 2.0, 2.5)
    expensive = affordability_factor(200_000.0, gdp, 2.0, 2.5)
    assert cheap > 0.95
    assert at_ref == pytest.approx(0.5, abs=1e-9)  # logistic is exactly 0.5 at ref
    assert expensive < 0.05
    assert cheap > at_ref > expensive


def test_affordability_shrinks_unreimbursed_value_more_for_expensive_drug():
    # Same market, no reimbursement, guaranteed launch: the fraction of value
    # retained out-of-pocket must be lower for a much more expensive drug.
    from valuation_engine.inputs.packs import load_modality, load_territory
    from valuation_engine.rnpv import run_model

    mod = load_modality("biologic")
    terr = load_territory("mexico")
    loa = SourcedValue(value=1.0, kind="bernoulli")

    def retained_fraction(price):
        asset = _biologic_asset(global_list_price_usd=price)
        r = resolve(asset, mod, terr, loa)
        base = r.base_params()
        full = run_model(r, {**base, "p_territory_approval": 1.0}).pie_rnpv
        noreimb = run_model(r, {**base, "p_reimbursement": 0.0, "p_territory_approval": 1.0}).pie_rnpv
        return noreimb / full

    assert retained_fraction(2_000) > retained_fraction(200_000)


def test_tiered_royalty_known_answer():
    net_sales = np.array([0.0, 30e6, 80e6])
    tiers = [
        RoyaltyTier(min_sales_usd=0, max_sales_usd=50e6, rate=0.10),
        RoyaltyTier(min_sales_usd=50e6, rate=0.15),
    ]
    r = royalty_by_year(net_sales, tiers)
    assert r[0] == 0.0
    assert r[1] == pytest.approx(30e6 * 0.10)
    assert r[2] == pytest.approx(50e6 * 0.10 + 30e6 * 0.15)


# --------------------------------------------------------------------------- #
# Assembler-level fixtures and invariants
# --------------------------------------------------------------------------- #
def _territory(**kw):
    defaults = dict(
        id="t", name="T", iso3="TTT", population=100_000_000,
        macro=MacroBlock(gdp_per_capita_usd=10000, health_exp_per_capita_usd=1000, pct_gov_health=0.4),
        base_discount_rate=0.09, country_risk_premium=0.03,
        regulatory_review_years=1.0, reimbursement_lag_years=1.0,
        p_reimbursement_default=0.6, private_channel_share=0.15,
        public_gtn_discount=0.3, private_gtn_discount=0.1,
        reference_price_factor=0.6, annual_price_erosion=0.03,
        filing_cost_usd=500_000, market_access_spend_usd=2_000_000,
    )
    defaults.update(kw)
    return TerritoryPack(**defaults)


def _biologic_asset(**kw):
    defaults = dict(
        id="a", name="A", modality_id="biologic", indication="X", phase="phase3",
        epidemiology=EpidemiologyBlock(basis="prevalence", rate_per_100k=400, diagnosis_rate=0.6, treatment_rate=0.5, eligible_fraction=0.2),
        global_list_price_usd=20000, exclusivity_years=10, peak_share=0.25,
    )
    defaults.update(kw)
    return AssetSpec(**defaults)


def test_deal_split_reconciles_to_pie():
    modality = ModalityPack(id="biologic", name="Biologic", dosing="recurring", default_cogs_pct=0.15)
    deal = DealTerms(
        upfront_usd=2e6,
        milestones=[Milestone(label="appr", amount_usd=5e6, trigger="on_approval")],
        royalty_tiers=[RoyaltyTier(min_sales_usd=0, rate=0.12)],
    )
    r = resolve(_biologic_asset(), modality, _territory(), SourcedValue(value=0.6, kind="bernoulli"), deal)
    res = evaluate_base(r)
    assert res.deal.licensor_value + res.deal.licensee_value == pytest.approx(res.pie_rnpv)


def test_launch_weight_scales_sales():
    modality = ModalityPack(id="biologic", name="Biologic", dosing="recurring", default_cogs_pct=0.15)
    asset = _biologic_asset()
    # certain launch
    r_hi = resolve(asset, modality, _territory(), SourcedValue(value=1.0, kind="bernoulli"))
    # override territory approval to 1 as well by editing params post-resolve
    res_hi = evaluate_base(r_hi)
    r_lo = resolve(asset, modality, _territory(), SourcedValue(value=0.5, kind="bernoulli"))
    res_lo = evaluate_base(r_lo)
    assert res_hi.net_sales.max() > res_lo.net_sales.max()


def test_gene_therapy_drawdown_mass_balance():
    modality = ModalityPack(id="gt", name="Gene therapy", dosing="one_time", default_cogs_pct=0.3,
                            one_time_annual_replenishment=0.03)
    asset = _biologic_asset(id="gtx", modality_id="gt", global_list_price_usd=2_000_000,
                            peak_share=0.6, exclusivity_years=12)
    territory = _territory(horizon_years=15)
    r = resolve(asset, modality, territory, SourcedValue(value=1.0, kind="bernoulli"))
    res = run_model(r, {**r.base_params(), "p_territory_approval": 1.0})
    base_elig = 100_000_000 * (400 / 1e5) * 0.6 * 0.5 * 0.2  # prevalent bolus
    max_available = base_elig + 0.03 * base_elig * territory.horizon_years
    assert res.treated_patients.sum() <= max_available + 1e-6


def test_one_time_is_more_front_loaded_than_recurring():
    # Same epi; one_time should peak earlier and decline; recurring stays flat-ish.
    territory = _territory(horizon_years=15)
    loa = SourcedValue(value=1.0, kind="bernoulli")
    rec_mod = ModalityPack(id="biologic", name="B", dosing="recurring", default_cogs_pct=0.2)
    ot_mod = ModalityPack(id="gt", name="G", dosing="one_time", default_cogs_pct=0.3)
    rec = run_model(resolve(_biologic_asset(), rec_mod, territory, loa), {**resolve(_biologic_asset(), rec_mod, territory, loa).base_params(), "p_territory_approval": 1.0})
    ot_asset = _biologic_asset(id="g", modality_id="gt")
    ot_res = run_model(resolve(ot_asset, ot_mod, territory, loa), {**resolve(ot_asset, ot_mod, territory, loa).base_params(), "p_territory_approval": 1.0})
    # one_time treated share of total that lands in the first 5 years is higher
    rec_frac_early = rec.treated_patients[:5].sum() / rec.treated_patients.sum()
    ot_frac_early = ot_res.treated_patients[:5].sum() / ot_res.treated_patients.sum()
    assert ot_frac_early > rec_frac_early
