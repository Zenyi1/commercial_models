"""Tests for the Valyu DeepSearch provider — parsing + extraction against a
saved fixture, and the provider end-to-end with an injected fake client (no
network), mirroring tests/test_edgar.py."""

import json
from pathlib import Path

import pytest

from valuation_engine.research.provider import ResearchQuery
from valuation_engine.research.valyu_provider import (
    ValyuProvider,
    ValyuResult,
    _confidence_from_relevance,
    build_query,
    extract_sourced_value,
    parse_results,
)

FIXTURE = Path(__file__).parent / "fixtures" / "valyu_deepsearch.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIXTURE.read_text())


def test_parse_results_sorted_by_relevance(payload):
    results = parse_results(payload)
    assert len(results) == 3
    assert [round(r.relevance, 2) for r in results] == [0.91, 0.83, 0.71]
    assert results[0].url == "https://www.gov.br/conitec/report-2023-gout"


def test_build_query_uses_indication_and_territory():
    q = ResearchQuery(key="p_reimbursement", territory_id="brazil", indication="gout")
    text = build_query(q)
    assert "reimbursed" in text and "Brazil" in text and "gout" in text


def test_extract_probability_reimbursement(payload):
    results = parse_results(payload)
    sv = extract_sourced_value("p_reimbursement", results)
    assert sv is not None
    assert sv.value == pytest.approx(0.45)
    assert sv.kind == "bernoulli"
    assert sv.provenance.source == "https://www.gov.br/conitec/report-2023-gout"
    assert "45%" in sv.provenance.quote
    # Graded from Valyu's relevance_score (0.91 -> high), not a hardcoded floor.
    assert sv.provenance.confidence == "high"


def test_extract_reimbursement_lag_years(payload):
    results = parse_results(payload)
    sv = extract_sourced_value("reimbursement_lag_years", results)
    assert sv is not None
    assert sv.value == pytest.approx(2.0)
    assert sv.unit == "years"


def test_extract_epi_rate_per_100k(payload):
    results = parse_results(payload)
    sv = extract_sourced_value("epi_rate_per_100k", results)
    assert sv is not None
    assert sv.value == pytest.approx(2600.0)


def test_extract_treatment_rate(payload):
    results = parse_results(payload)
    sv = extract_sourced_value("treatment_rate", results)
    assert sv is not None
    assert 0.0 < sv.value <= 1.0
    assert sv.value == pytest.approx(0.40)


def test_extract_market_access_spend_usd(payload):
    results = parse_results(payload)
    sv = extract_sourced_value("market_access_spend_usd", results)
    assert sv is not None
    assert sv.value == pytest.approx(3_500_000.0)
    # Pulled from the 0.71-relevance result -> medium (not high, not the old low).
    assert sv.provenance.confidence == "medium"


def test_confidence_graded_from_relevance():
    assert _confidence_from_relevance(0.95) == "high"
    assert _confidence_from_relevance(0.80) == "high"
    assert _confidence_from_relevance(0.70) == "medium"
    assert _confidence_from_relevance(0.60) == "medium"
    assert _confidence_from_relevance(0.40) == "low"


def test_extract_returns_none_when_absent():
    # No sentence mentions peak share -> must not fabricate.
    results = [ValyuResult(title="t", url="u", content="Gout is common.", source="s", relevance=0.9)]
    assert extract_sourced_value("peak_share", results) is None


def test_peak_share_not_confused_by_market_access_dollars(payload):
    # Regression: "market-access investment ... $3.5 million" must NOT yield a
    # peak_share of 0.5 (the "market" hint + the ".5" in "3.5" used to collide).
    results = parse_results(payload)
    assert extract_sourced_value("peak_share", results) is None


def test_fraction_ignores_decimal_inside_larger_number():
    results = [ValyuResult(title="t", url="https://x", content="Spend was about $3.5 million on the launch.",
                           source="x", relevance=0.9)]
    # eligible_fraction has no matching sentence here -> None, not 0.5.
    assert extract_sourced_value("eligible_fraction", results) is None


def test_provider_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("VALYU_API_KEY", raising=False)
    assert ValyuProvider(api_key=None).available() is False
    assert ValyuProvider(api_key=None).get(
        ResearchQuery(key="p_reimbursement", territory_id="brazil")
    ) is None


def test_provider_end_to_end_with_fake_client(payload):
    class _FakeClient:
        def deepsearch(self, query, **kw):
            return payload

    prov = ValyuProvider(api_key="test-key")
    prov._make_client = lambda: _FakeClient()  # inject stub, no network
    sv = prov.get(ResearchQuery(key="p_reimbursement", territory_id="brazil", indication="gout"))
    assert sv is not None and sv.value == pytest.approx(0.45)


def test_provider_swallows_backend_errors():
    class _BoomClient:
        def deepsearch(self, query, **kw):
            raise OSError("network down")

    prov = ValyuProvider(api_key="test-key")
    prov._make_client = lambda: _BoomClient()
    # Must degrade to None, never raise, so the aggregator can fall back.
    assert prov.get(ResearchQuery(key="p_reimbursement", territory_id="brazil")) is None
