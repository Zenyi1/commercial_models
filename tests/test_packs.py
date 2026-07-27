"""Validate that all built-in territory and modality packs load and are usable."""

import pytest

from valuation_engine.inputs.packs import (
    available_modalities,
    available_territories,
    load_modality,
    load_territory,
)
from valuation_engine.inputs.resolve import resolve
from valuation_engine.inputs.schemas import AssetSpec, EpidemiologyBlock, SourcedValue
from valuation_engine.rnpv import evaluate_base

EXPECTED_TERRITORIES = {"mexico", "brazil", "saudi_arabia"}
EXPECTED_MODALITIES = {"small_molecule", "biologic", "gene_therapy"}


def test_expected_packs_present():
    assert EXPECTED_TERRITORIES <= set(available_territories())
    assert EXPECTED_MODALITIES <= set(available_modalities())


@pytest.mark.parametrize("tid", sorted(EXPECTED_TERRITORIES))
def test_territory_pack_loads(tid):
    t = load_territory(tid)
    assert t.id == tid
    assert t.population.base() > 0
    # discount rate is positive and sane
    rate = t.base_discount_rate.base() + t.country_risk_premium.base()
    assert 0.05 < rate < 0.25


@pytest.mark.parametrize("mid", sorted(EXPECTED_MODALITIES))
def test_modality_pack_loads(mid):
    m = load_modality(mid)
    assert m.id == mid
    assert 0 <= m.default_cogs_pct.base() <= 1


def test_every_territory_runs_end_to_end():
    mod = load_modality("biologic")
    asset = AssetSpec(
        id="a", name="A", modality_id="biologic", indication="X", phase="phase3",
        epidemiology=EpidemiologyBlock(basis="prevalence", rate_per_100k=500,
                                       diagnosis_rate=0.6, treatment_rate=0.5, eligible_fraction=0.2),
        global_list_price_usd=SourcedValue(value=20000), exclusivity_years=10, peak_share=0.25,
    )
    loa = SourcedValue(value=0.6, kind="bernoulli")
    for tid in sorted(EXPECTED_TERRITORIES):
        res = evaluate_base(resolve(asset, mod, load_territory(tid), loa))
        assert res.pie_rnpv > 0
        assert res.net_value < res.pie_rnpv  # cost-to-enter subtracted
