"""Tests for the clinical plug-in and the research/evidence layer."""

import pytest

from valuation_engine.clinical.provider import DefaultLoAProvider
from valuation_engine.inputs.schemas import AssetSpec, EpidemiologyBlock, SourcedValue
from valuation_engine.research.cached_provider import CachedProvider
from valuation_engine.research.enrich import enrich_territory_asset
from valuation_engine.research.provider import MultiSourceProvider, ResearchProvider, ResearchQuery


def _asset(**kw):
    d = dict(
        id="a", name="A", modality_id="biologic", indication="X", phase="phase3",
        epidemiology=EpidemiologyBlock(basis="prevalence", rate_per_100k=400,
                                       diagnosis_rate=0.6, treatment_rate=0.5, eligible_fraction=0.2),
        global_list_price_usd=20000, exclusivity_years=10, peak_share=0.25,
    )
    d.update(kw)
    return AssetSpec(**d)


# --------------------------------------------------------------------------- #
# Clinical plug-in
# --------------------------------------------------------------------------- #
def test_default_loa_is_bernoulli_and_phase_ordered():
    prov = DefaultLoAProvider()
    p1 = prov.probability_of_launch(_asset(phase="phase1"))
    p3 = prov.probability_of_launch(_asset(phase="phase3"))
    assert p1.kind == "bernoulli" and p3.kind == "bernoulli"
    assert p3.value > p1.value  # later phase -> higher LoA
    assert p3.provenance.confidence == "low"  # flagged as placeholder


def test_asset_override_wins():
    prov = DefaultLoAProvider()
    override = SourcedValue(value=0.42, kind="bernoulli")
    sv = prov.probability_of_launch(_asset(clinical_loa_override=override))
    assert sv.value == 0.42


# --------------------------------------------------------------------------- #
# Cached provider + enrichment
# --------------------------------------------------------------------------- #
def _cached():
    return CachedProvider({
        "assets": {
            "a": {"brazil": {
                "p_reimbursement": {"value": 0.7, "kind": "bernoulli",
                                     "provenance": {"source": "http://example/hta", "confidence": "medium"}},
                "reference_price_factor": {"value": 0.5,
                                            "provenance": {"confidence": "medium"}},
            }}
        },
        "territory_defaults": {
            "brazil": {"launch_delay_years": {"value": 1.0, "provenance": {"confidence": "low"}}}
        },
    })


def test_cached_provider_asset_then_default_fallback():
    prov = _cached()
    got = prov.get(ResearchQuery(key="p_reimbursement", territory_id="brazil", asset_id="a"))
    assert got is not None and got.value == 0.7
    # Falls back to territory default when no asset-specific value.
    fb = prov.get(ResearchQuery(key="launch_delay_years", territory_id="brazil", asset_id="a"))
    assert fb is not None and fb.value == 1.0
    # Unknown key -> None.
    assert prov.get(ResearchQuery(key="peak_share", territory_id="brazil", asset_id="a")) is None


def test_enrich_builds_overrides_from_provider():
    tai = enrich_territory_asset(_cached(), _asset(), "brazil")
    assert tai.p_reimbursement is not None and tai.p_reimbursement.value == 0.7
    assert tai.reference_price_factor is not None
    assert tai.peak_share is None  # provider didn't answer it


# --------------------------------------------------------------------------- #
# Multi-source cross-validation
# --------------------------------------------------------------------------- #
class _Fixed(ResearchProvider):
    def __init__(self, name, value, confidence):
        self.name = name
        self._v = SourcedValue(value=value, provenance={"confidence": confidence})

    def get(self, query):
        return self._v


def test_multisource_prefers_higher_confidence_and_flags_disagreement():
    lo = _Fixed("lo", 0.5, "low")
    hi = _Fixed("hi", 0.9, "high")
    multi = MultiSourceProvider([lo, hi])
    got = multi.get(ResearchQuery(key="reference_price_factor", territory_id="brazil"))
    assert got.value == 0.9  # higher confidence wins
    # 0.5 vs 0.9 is a large relative gap -> confidence downgraded + noted.
    assert got.provenance.confidence == "low"
    assert "disagreement" in (got.provenance.notes or "")


def test_multisource_agreement_keeps_confidence():
    a = _Fixed("a", 0.90, "high")
    b = _Fixed("b", 0.88, "medium")
    multi = MultiSourceProvider([a, b])
    got = multi.get(ResearchQuery(key="reference_price_factor", territory_id="brazil"))
    assert got.value == 0.90
    assert got.provenance.confidence == "high"  # close enough, not downgraded
