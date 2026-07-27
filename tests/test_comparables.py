"""Tests for the comparables data foundation and triangulation."""

import pytest

from valuation_engine.comparables.analyzer import ComparablesAnalyzer
from valuation_engine.comparables.harvest import (
    FileDealsSource,
    default_corpus_path,
    harvest,
    load_corpus,
)
from valuation_engine.comparables.triangulation import triangulate
from valuation_engine.inputs.packs import load_territory


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(default_corpus_path())


@pytest.fixture(scope="module")
def analyzer(corpus):
    return ComparablesAnalyzer(corpus)


def test_corpus_loads_and_validates(corpus):
    assert len(corpus) >= 12
    assert all(0 <= r.peak_royalty_rate <= 1 for r in corpus if r.peak_royalty_rate)


def test_harvest_dedupes_identical_sources():
    src = FileDealsSource(default_corpus_path())
    res = harvest([src, FileDealsSource(default_corpus_path())])
    n = len(load_corpus(default_corpus_path()))
    assert len(res.records) == n  # deduped
    assert res.duplicates_dropped == n  # the second copy fully dropped


def test_royalty_benchmark_percentiles_ordered(analyzer):
    b = analyzer.royalty_benchmark(modality="biologic")
    assert b.n >= 1
    assert b.p25 <= b.p50 <= b.p75


def test_filter_by_region_and_stage(analyzer):
    latam = analyzer.filter(region="latam")
    assert len(latam) >= 1 and all("latam" in r.regions for r in latam)


def test_geography_factor_orders_by_market_size(analyzer):
    br = analyzer.geography_adjustment_factor(load_territory("brazil"))
    mx = analyzer.geography_adjustment_factor(load_territory("mexico"))
    sa = analyzer.geography_adjustment_factor(load_territory("saudi_arabia"))
    for f in (br, mx, sa):
        assert 0 < f < 1
    assert br > mx  # Brazil is the larger market by the proxy
    assert br > sa


def test_territorial_split_prior_in_unit_interval(analyzer):
    for region in ("latam", "mena", "apac"):
        p = analyzer.territorial_split_prior(region)
        assert 0 < p < 1


def test_implied_value_scales_with_reference(analyzer):
    t = load_territory("brazil")
    v1 = analyzer.implied_territory_value(t, 100e6)
    v2 = analyzer.implied_territory_value(t, 200e6)
    assert v2 == pytest.approx(2 * v1)


def test_triangulation_flags_divergence(analyzer):
    t = load_territory("brazil")
    # Bottom-up wildly higher than the other legs -> flagged.
    tri = triangulate(500e6, 5e6, analyzer, t, reference_value_usd=50e6)
    assert tri.flagged
    assert any("divergence" in n for n in tri.notes)


def test_triangulation_agreement_not_flagged():
    # Two consistent legs (bottom-up ~= top-down), no comparables leg.
    tri = triangulate(50e6, 20e6)  # top_down = 20*2.5 = 50
    assert not tri.flagged
    assert tri.reconciled == pytest.approx(50e6)
