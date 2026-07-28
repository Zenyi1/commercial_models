"""Valyu DeepSearch backend — live retrieval + sourced-value extraction.

Two layers, kept separate so the parsing/extraction is testable without the
network (mirrors :mod:`valuation_engine.research.data_sources.edgar`):

* :class:`ValyuClient` — a thin, self-rate-limited ``urllib`` wrapper over the
  Valyu DeepSearch REST API. No third-party SDK; stdlib only.
* pure functions — :func:`build_query` turns a :class:`ResearchQuery` into a
  natural-language search; :func:`parse_results` turns the JSON payload into
  :class:`ValyuResult`s; :func:`extract_sourced_value` turns those results into a
  :class:`SourcedValue` carrying a real source URL and a verbatim quote, or
  ``None`` when no confident number is found.

:class:`ValyuProvider` ties them together. It is *available* only when
``VALYU_API_KEY`` is set. Extraction is deliberately conservative: it returns a
value only when a number of the expected shape appears near the right keywords,
tags it ``confidence="low"`` (machine-extracted, human should verify against the
quote), and otherwise returns ``None`` — it never fabricates a point estimate.

Injecting a stronger extractor (e.g. an LLM) is a one-argument change: pass
``extractor=`` to :class:`ValyuProvider`; the client/query layers are unchanged.

--------------------------------------------------------------------------------
API surface verified against docs.valyu.ai (2026-07): ``POST
https://api.valyu.ai/v1/search`` with an ``x-api-key`` header. Endpoint/auth are
isolated as module constants and the response field names live in one place
(:func:`parse_results`), so any future schema change is a one-line fix.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from valuation_engine.inputs.schemas import Provenance, SourcedValue
from valuation_engine.research.provider import ResearchProvider, ResearchQuery

# -- Valyu API surface (verified against docs.valyu.ai, 2026-07) -------------- #
# POST https://api.valyu.ai/v1/search  — header x-api-key, JSON body
# {query, search_type: all|web|proprietary, max_num_results: 1-20,
#  relevance_threshold: 0-1}; response has a `results` array of objects with
# title/url/content/source/relevance_score.
_DEEPSEARCH_URL = "https://api.valyu.ai/v1/search"
_AUTH_HEADER = "x-api-key"
_DEFAULT_SEARCH_TYPE = "all"          # "all" | "web" | "proprietary"
_DEFAULT_MAX_RESULTS = 8
_DEFAULT_RELEVANCE = 0.5

# Human territory names for query phrasing; unknown ids fall back to the id.
_TERRITORY_NAMES = {
    "mexico": "Mexico",
    "brazil": "Brazil",
    "saudi_arabia": "Saudi Arabia",
}


@dataclass(frozen=True)
class ValyuResult:
    title: str
    url: str
    content: str
    source: str
    relevance: float


# --------------------------------------------------------------------------- #
# Query building (pure)
# --------------------------------------------------------------------------- #
def build_query(q: ResearchQuery) -> str:
    """Natural-language search for one canonical key × asset × territory."""
    terr = _TERRITORY_NAMES.get(q.territory_id, q.territory_id.replace("_", " "))
    ind = q.indication or "the drug"
    templates = {
        "epi_rate_per_100k": f"{ind} prevalence or incidence per 100,000 population in {terr}",
        "diagnosis_rate": f"proportion of {ind} patients diagnosed in {terr}",
        "treatment_rate": f"share of diagnosed {ind} patients receiving drug therapy in {terr}",
        "eligible_fraction": f"fraction of treated {ind} patients eligible for a second-line branded therapy",
        "p_territory_approval": f"regulatory approval status and likelihood for {ind} drugs in {terr}",
        "p_reimbursement": f"probability a branded {ind} drug is reimbursed on the public formulary in {terr}",
        "reimbursement_lag_years": f"time in years from drug approval to public reimbursement listing in {terr}",
        "regulatory_review_years": f"average drug regulatory review time in years in {terr}",
        "net_price_usd": f"annual net price in USD of a branded {ind} drug in {terr}",
        "reference_price_factor": f"drug prices in {terr} as a fraction of US or global reference price",
        "peak_share": f"peak market share of a branded {ind} drug in {terr}",
        "launch_delay_years": f"typical launch delay in years for new drugs in {terr}",
        "market_access_spend_usd": f"pharmaceutical product launch and market-access spend in USD in {terr}",
    }
    return templates.get(q.key, f"{q.key.replace('_', ' ')} for {ind} in {terr}")


# --------------------------------------------------------------------------- #
# Response parsing (pure)
# --------------------------------------------------------------------------- #
def parse_results(payload: dict) -> list[ValyuResult]:
    """Parse a Valyu DeepSearch JSON payload into results (no I/O)."""
    out: list[ValyuResult] = []
    for r in payload.get("results", []) or []:
        content = r.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        out.append(ValyuResult(
            title=str(r.get("title", "")),
            url=str(r.get("url", "")),
            content=content,
            source=str(r.get("source", "")),
            relevance=float(r.get("relevance_score", 0.0) or 0.0),
        ))
    out.sort(key=lambda x: x.relevance, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Numeric extraction (pure, heuristic, conservative)
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# Split a sentence into clauses on separators that are FOLLOWED BY WHITESPACE, so
# thousands separators inside a number ("2,600 per 100,000") are never broken —
# only clause boundaries like "..diagnosed, and of those.." are. This lets a
# sentence with two figures ("50% diagnosed ... 40% treated") map each figure to
# the clause that describes it.
_CLAUSE_SPLIT = re.compile(r",\s+|;\s+|\s+and\s+|\s+but\s+|\s+while\s+|\s+whereas\s+", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_PER_100K = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:cases\s+)?per\s+100[,\s]?000", re.I)
_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*(year|yr|month|mo)s?\b", re.I)
_MONEY = re.compile(
    r"\$?\s*(\d+(?:[.,]\d+)?)\s*(billion|million|thousand|bn|m|k)?\b", re.I
)
_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6,
          "thousand": 1e3, "k": 1e3}

# Per-key: how to read a number, plausibility bounds, and the resulting kind.
_PROBABILITY_KEYS = {"p_territory_approval", "p_reimbursement"}
_FRACTION_KEYS = {"diagnosis_rate", "treatment_rate", "eligible_fraction",
                  "reference_price_factor", "peak_share"}
_YEAR_KEYS = {"reimbursement_lag_years", "regulatory_review_years", "launch_delay_years"}
_USD_KEYS = {"net_price_usd", "market_access_spend_usd"}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _clauses(sentence: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]


def _first_fraction(sentence: str) -> Optional[float]:
    m = _PCT.search(sentence)
    if m:
        return float(m.group(1)) / 100.0
    # bare 0..1 decimal ("a probability of 0.65"). The lookbehind stops it from
    # grabbing the ".5" out of a larger number like "$3.5 million".
    m = re.search(r"(?<![\d.])(0?\.\d+)\b", sentence)
    return float(m.group(1)) if m else None


def _first_years(sentence: str) -> Optional[float]:
    m = _YEARS.search(sentence)
    if not m:
        return None
    val = float(m.group(1))
    return val / 12.0 if m.group(2).lower().startswith("mo") else val


def _first_per_100k(sentence: str) -> Optional[float]:
    m = _PER_100K.search(sentence)
    if m:
        return float(m.group(1).replace(",", ""))
    # a prevalence percentage → convert to per-100k
    m = _PCT.search(sentence)
    if m:
        return float(m.group(1)) / 100.0 * 1e5
    return None


def _first_usd(sentence: str) -> Optional[float]:
    m = _MONEY.search(sentence)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    scale = _SCALE.get((m.group(2) or "").lower(), 1.0)
    return val * scale


def _value_for_key(key: str, sentence: str) -> Optional[tuple[float, str, str]]:
    """Return ``(value, kind, unit)`` extracted for ``key`` from ``sentence``."""
    if key in _PROBABILITY_KEYS:
        v = _first_fraction(sentence)
        if v is not None and 0.0 <= v <= 1.0:
            return v, "bernoulli", "probability"
    elif key in _FRACTION_KEYS:
        v = _first_fraction(sentence)
        if v is not None and 0.0 <= v <= 1.0:
            return v, "point", "fraction"
    elif key in _YEAR_KEYS:
        v = _first_years(sentence)
        if v is not None and 0.0 <= v <= 30.0:
            return v, "point", "years"
    elif key == "epi_rate_per_100k":
        v = _first_per_100k(sentence)
        if v is not None and 0.0 < v <= 1e5:
            return v, "point", "per_100k"
    elif key in _USD_KEYS:
        v = _first_usd(sentence)
        if v is not None and v > 0.0:
            return v, "point", "usd"
    return None


# Keywords a sentence should contain to be a credible answer for the key.
_KEY_HINTS = {
    "p_reimbursement": ("reimburs", "formulary", "cover"),
    "p_territory_approval": ("approv", "authoriz"),
    "reimbursement_lag_years": ("reimburs", "listing", "access"),
    "regulatory_review_years": ("review", "approv", "regulator"),
    "launch_delay_years": ("launch", "delay"),
    "net_price_usd": ("price", "cost", "annual"),
    "reference_price_factor": ("reference price", "of us price", "of the us", "reference-price"),
    "peak_share": ("share",),  # NOT "market" — that matches "market-access" etc.
    "market_access_spend_usd": ("launch", "market access", "spend", "investment"),
    "epi_rate_per_100k": ("prevalen", "inciden", "per 100"),
    "diagnosis_rate": ("diagnos",),
    "treatment_rate": ("treat", "therapy"),
    "eligible_fraction": ("eligib", "second-line", "second line"),
}


def extract_sourced_value(key: str, results: Sequence[ValyuResult]) -> Optional[SourcedValue]:
    """Best-effort sourced value for ``key`` from Valyu results, or ``None``.

    Conservative by design: requires a number of the expected shape in a
    sentence that also mentions the key's topic, keeps the verbatim sentence as
    the quote and the result URL as the source, and marks confidence ``low``
    (machine-extracted). Never fabricates.
    """
    hints = _KEY_HINTS.get(key, ())
    for r in results:  # already sorted by relevance
        for sentence in _sentences(r.content):
            for clause in _clauses(sentence):
                low = clause.lower()
                if hints and not any(h in low for h in hints):
                    continue
                got = _value_for_key(key, clause)
                if got is None:
                    continue
                value, kind, unit = got
                return SourcedValue(
                    value=value,
                    kind=kind,
                    unit=unit,
                    provenance=Provenance(
                        source=r.url or None,
                        quote=sentence[:400],  # full sentence for human verification
                        publisher=r.source or None,
                        method="valyu deepsearch (machine-extracted; verify against quote)",
                        confidence="low",
                    ),
                )
    return None


# --------------------------------------------------------------------------- #
# HTTP client (thin, rate-limited, stdlib only)
# --------------------------------------------------------------------------- #
class ValyuClient:
    """Minimal Valyu DeepSearch client. No SDK — stdlib ``urllib`` only."""

    def __init__(self, api_key: str, min_interval_s: float = 0.2, timeout_s: float = 30.0):
        if not api_key:
            raise ValueError("Valyu API key required")
        self.api_key = api_key
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()

    def deepsearch(
        self,
        query: str,
        max_num_results: int = _DEFAULT_MAX_RESULTS,
        relevance_threshold: float = _DEFAULT_RELEVANCE,
        search_type: str = _DEFAULT_SEARCH_TYPE,
    ) -> dict:
        body = json.dumps({
            "query": query,
            "search_type": search_type,
            "max_num_results": max_num_results,
            "relevance_threshold": relevance_threshold,
        }).encode("utf-8")
        self._throttle()
        req = urllib.request.Request(
            _DEEPSEARCH_URL,
            data=body,
            method="POST",
            headers={
                _AUTH_HEADER: self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
Extractor = Callable[[str, Sequence[ValyuResult]], Optional[SourcedValue]]


class ValyuProvider(ResearchProvider):
    name = "valyu"

    def __init__(
        self,
        api_key: Optional[str] = None,
        extractor: Optional[Extractor] = None,
        max_num_results: int = _DEFAULT_MAX_RESULTS,
    ):
        self.api_key = api_key or os.getenv("VALYU_API_KEY")
        self.extractor: Extractor = extractor or extract_sourced_value
        self.max_num_results = max_num_results

    def available(self) -> bool:
        return bool(self.api_key)

    def _make_client(self) -> ValyuClient:
        # Seam for tests: patch this to inject a fake client.
        return ValyuClient(self.api_key or "")

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        if not self.available():
            return None
        return self._research(query)

    def _research(self, query: ResearchQuery) -> Optional[SourcedValue]:
        try:
            payload = self._make_client().deepsearch(
                build_query(query), max_num_results=self.max_num_results
            )
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            # A flaky/unauthorized backend must not sink the whole valuation;
            # the aggregator falls back to other providers / cached evidence.
            return None
        results = parse_results(payload)
        if not results:
            return None
        return self.extractor(query.key, results)
