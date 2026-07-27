"""Tests for Monte Carlo simulation and tornado sensitivity."""

import numpy as np
import pytest

from valuation_engine.core.montecarlo import simulate, tornado
from valuation_engine.inputs.resolve import resolve
from valuation_engine.inputs.schemas import (
    AssetSpec,
    EpidemiologyBlock,
    MacroBlock,
    ModalityPack,
    SourcedValue,
    TerritoryPack,
)


def _setup():
    modality = ModalityPack(id="biologic", name="Biologic", dosing="recurring", default_cogs_pct=0.15)
    territory = TerritoryPack(
        id="t", name="T", iso3="TTT", population=100_000_000,
        macro=MacroBlock(gdp_per_capita_usd=10000, health_exp_per_capita_usd=1000, pct_gov_health=0.4),
        base_discount_rate=0.09, country_risk_premium=0.03,
        regulatory_review_years=SourcedValue(value=1.5, low=1.0, high=2.5),
        reimbursement_lag_years=1.0,
        p_reimbursement_default=SourcedValue(value=0.55, kind="bernoulli"),
        private_channel_share=0.15, public_gtn_discount=0.3, private_gtn_discount=0.1,
        reference_price_factor=SourcedValue(value=0.6, low=0.45, high=0.75),
        annual_price_erosion=0.03, filing_cost_usd=500_000, market_access_spend_usd=2_000_000,
    )
    asset = AssetSpec(
        id="a", name="A", modality_id="biologic", indication="X", phase="phase3",
        epidemiology=EpidemiologyBlock(
            basis="prevalence",
            rate_per_100k=SourcedValue(value=400, low=300, high=550),
            diagnosis_rate=0.6, treatment_rate=0.5,
            eligible_fraction=SourcedValue(value=0.2, low=0.12, high=0.3),
        ),
        global_list_price_usd=SourcedValue(value=20000, low=15000, high=25000),
        exclusivity_years=10, peak_share=SourcedValue(value=0.25, low=0.15, high=0.35),
    )
    loa = SourcedValue(value=0.6, kind="bernoulli")
    return resolve(asset, modality, territory, loa)


def test_simulation_is_reproducible():
    r = _setup()
    a = simulate(r, n=2000, seed=99)
    b = simulate(r, n=2000, seed=99)
    assert np.array_equal(a.metrics["pie_rnpv"], b.metrics["pie_rnpv"])


def test_percentiles_are_ordered():
    r = _setup()
    s = simulate(r, n=4000, seed=1).summary("pie_rnpv")
    assert s.p10 <= s.p50 <= s.p90
    assert 0.0 <= s.p_positive <= 1.0


def test_binary_gates_produce_zero_mass():
    # With ~60% launch and ~55% reimbursement, a meaningful share of draws should
    # land at (near) zero value -> P10 should be ~0.
    r = _setup()
    s = simulate(r, n=5000, seed=3).summary("pie_rnpv")
    assert s.p10 == pytest.approx(0.0, abs=1.0)
    assert s.p90 > s.p50 > 0


def test_deal_split_metrics_reconcile_each_draw():
    r = _setup()
    sim = simulate(r, n=1000, seed=5)
    total = sim.metrics["licensor_value"] + sim.metrics["licensee_value"]
    assert np.allclose(total, sim.metrics["pie_rnpv"], rtol=1e-9, atol=1e-3)


def test_tornado_sorted_and_gates_dominate():
    r = _setup()
    tr = tornado(r, metric="pie_rnpv")
    swings = [b.swing for b in tr.bars]
    assert swings == sorted(swings, reverse=True)
    # The reimbursement/approval/clinical gates should be among the top drivers.
    top_params = {b.param for b in tr.bars[:4]}
    assert {"p_reimbursement", "p_territory_approval", "clinical_loa"} & top_params
