"""Tests for the SEC EDGAR deals source — parsing and extraction only (no network).

Live network calls are exercised by ``scripts/harvest_edgar.py``, not the test
suite, so tests stay deterministic and offline.
"""

import json
from pathlib import Path

import pytest

from valuation_engine.research.data_sources.edgar import (
    EdgarClient,
    SecEdgarDealsSource,
    classify_modality,
    classify_stage,
    extract_deal_terms,
    extract_regions,
    parse_search_results,
    total_hits,
)

FIXTURE = Path(__file__).parent / "fixtures" / "edgar_search.json"
EXCERPT = Path(__file__).parent / "fixtures" / "edgar_filing_excerpt.txt"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIXTURE.read_text())


def test_parse_search_results(payload):
    hits = parse_search_results(payload)
    assert len(hits) == 3
    h = hits[0]
    assert h.accession == "0001193125-23-002398"
    assert h.filename == "d405031d8k.htm"
    assert h.filer_name == "CytomX Therapeutics, Inc."  # symbol/CIK stripped
    assert h.form == "8-K"
    assert h.file_date == "2023-01-05"
    assert h.cik == "0001501989"


def test_document_url_construction(payload):
    h = parse_search_results(payload)[0]
    # leading zeros stripped from CIK, dashes stripped from accession
    assert h.document_url == (
        "https://www.sec.gov/Archives/edgar/data/1501989/"
        "000119312523002398/d405031d8k.htm"
    )


def test_total_hits(payload):
    assert total_hits(payload) == 175


def test_client_requires_contact_user_agent():
    with pytest.raises(ValueError):
        EdgarClient("no-contact-here")
    # a proper UA with an email is accepted
    EdgarClient("FirstOcean Research you@example.com")


# --------------------------------------------------------------------------- #
# Extraction against a real filing excerpt (CytomX / Moderna 8-K)
# --------------------------------------------------------------------------- #
def test_extract_deal_terms_from_real_excerpt():
    text = EXCERPT.read_text()
    terms = extract_deal_terms(text)
    assert terms["upfront_usd"] == 35_000_000.0
    # milestones "$1.2 billion" + upfront -> total biobucks
    assert terms["total_value_usd"] == pytest.approx(1_235_000_000.0)
    # royalty rate is redacted in this filing -> not fabricated
    assert terms["peak_royalty_rate"] is None


def test_classify_from_real_excerpt():
    text = EXCERPT.read_text()
    # "global net sales" -> global rights
    assert extract_regions(text) == ["global"]
    # The terms paragraph alone doesn't name the modality -> honestly "other".
    assert classify_modality(text) == "other"


def test_upfront_amount_can_precede_keyword():
    # Real failure mode: "$170 million in upfront payments and ... $3 billion in
    # milestones" — the amount precedes the keyword and a larger distractor
    # follows it. Nearest-money must pick $170M, not $3B.
    text = ("Mersana with $170 million in upfront payments and an opportunity "
            "for more than $3 billion in milestones")
    terms = extract_deal_terms(text)
    assert terms["upfront_usd"] == 170_000_000.0
    assert terms["total_value_usd"] == pytest.approx(170_000_000.0 + 3_000_000_000.0)


def test_royalty_range_takes_peak():
    text = "tiered royalties ranging from 8% to 15% on net sales"
    assert extract_deal_terms(text)["peak_royalty_rate"] == pytest.approx(0.15)


def test_single_royalty_rate():
    text = "a royalty of 12% on annual net sales"
    assert extract_deal_terms(text)["peak_royalty_rate"] == pytest.approx(0.12)


def test_modality_and_stage_keywords():
    assert classify_modality("an AAV gene therapy program") == "gene_therapy"
    assert classify_modality("a small molecule inhibitor") == "small_molecule"
    assert classify_stage("currently in Phase 3 development") == "phase3"
    assert classify_stage("a preclinical asset") == "preclinical"


def test_missing_terms_return_none():
    terms = extract_deal_terms("The parties entered into a collaboration agreement.")
    assert terms == {"upfront_usd": None, "total_value_usd": None, "peak_royalty_rate": None}


# --------------------------------------------------------------------------- #
# SecEdgarDealsSource end-to-end with a stubbed client (no network)
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self, hits, text):
        self._hits, self._text = hits, text

    def search(self, *a, **k):
        return self._hits

    def fetch_document_text(self, hit):
        return self._text


def test_source_unavailable_without_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    assert SecEdgarDealsSource(user_agent=None).available() is False
    assert SecEdgarDealsSource(user_agent=None).fetch() == []


def test_source_builds_deal_records(payload):
    hits = parse_search_results(payload)
    text = EXCERPT.read_text()
    src = SecEdgarDealsSource(user_agent="FirstOcean Research you@example.com")
    src._make_client = lambda: _FakeClient(hits, text)  # inject stub

    records = src.fetch()
    assert len(records) == 3
    r = records[0]
    # sourced to the real filing URL, terms extracted, not fabricated
    assert r.source.startswith("https://www.sec.gov/Archives/edgar/data/")
    assert r.upfront_usd == 35_000_000.0
    assert r.regions == ["global"] and r.deal_type == "global_license"
    assert r.licensor == "CytomX Therapeutics, Inc."
    assert r.confidence == "medium"  # had extractable terms


def test_source_records_are_valid_for_analyzer(payload):
    from valuation_engine.comparables.analyzer import ComparablesAnalyzer
    hits = parse_search_results(payload)
    src = SecEdgarDealsSource(user_agent="FirstOcean Research you@example.com")
    src._make_client = lambda: _FakeClient(hits, EXCERPT.read_text())
    an = ComparablesAnalyzer(src.fetch())
    b = an.upfront_benchmark()
    assert b.n >= 1 and b.p50 is not None
