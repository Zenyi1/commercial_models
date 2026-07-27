"""Tests for the SEC EDGAR deals source — parsing and extraction only (no network).

Live network calls are exercised by ``scripts/harvest_edgar.py``, not the test
suite, so tests stay deterministic and offline.
"""

import json
from pathlib import Path

import pytest

from valuation_engine.research.data_sources.edgar import (
    EdgarClient,
    parse_search_results,
    total_hits,
)

FIXTURE = Path(__file__).parent / "fixtures" / "edgar_search.json"


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
