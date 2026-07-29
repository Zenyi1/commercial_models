"""LLM extraction layer — read Valyu's passages, return a structured SourcedValue.

Valyu is a good *retriever* (relevant, cited passages) but a poor source of clean
numbers: real prose says "the process typically takes about two years", not a bare
"2". The regex extractor in :mod:`valyu_provider` misses those. This module is the
RAG *reader*: it hands the retrieved passages to an LLM and asks for one calibrated
number with its supporting quote and source URL — or ``found: false`` when the
passages don't contain it (never fabricated).

It plugs into the existing ``extractor=`` seam:

    from valuation_engine.research.valyu_provider import ValyuProvider
    from valuation_engine.research.llm_extractor import make_llm_extractor
    provider = ValyuProvider(extractor=make_llm_extractor())

Stdlib only — a thin ``urllib`` client over the Anthropic Messages API (POST
/v1/messages), mirroring how the Valyu client avoids an SDK dependency. Structured
output is enforced with ``output_config.format`` (json_schema), so the model must
return exactly the fields we parse.

Model: defaults to ``claude-opus-4-8`` (override with ``ANTHROPIC_MODEL`` or the
``model=`` argument). Extraction is a light task, so ``claude-haiku-4-5`` is a
reasonable, much cheaper/faster choice — set ``ANTHROPIC_MODEL=claude-haiku-4-5``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional, Sequence

from valuation_engine.inputs.schemas import Provenance, SourcedValue
from valuation_engine.research.valyu_provider import (
    ValyuResult,
    _FRACTION_KEYS,
    _PROBABILITY_KEYS,
    _USD_KEYS,
    _YEAR_KEYS,
)

# -- Anthropic API surface (verify against docs; isolated here) --------------- #
_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-opus-4-8"  # per Anthropic guidance; ANTHROPIC_MODEL overrides
_MAX_RESULTS_TO_LLM = 5
_MAX_CHARS_PER_PASSAGE = 1500

# Plausibility bounds per key-type — the LLM's answer must land in range or we
# drop it (same discipline as the regex extractor: never emit a nonsense number).
_UNIT_BOUNDS = {
    "probability": (0.0, 1.0),
    "fraction": (0.0, 1.0),
    "years": (0.0, 30.0),
    "per_100k": (0.0, 1e5),
    "usd": (0.0, float("inf")),
}

# What we ask the LLM to find, per canonical key.
_KEY_ASK = {
    "epi_rate_per_100k": "the prevalence or incidence of the indication per 100,000 population",
    "diagnosis_rate": "the fraction (0-1) of patients with the indication who are diagnosed",
    "treatment_rate": "the fraction (0-1) of diagnosed patients who receive drug therapy",
    "eligible_fraction": "the fraction (0-1) of treated patients eligible for a branded second-line therapy",
    "p_territory_approval": "the probability (0-1) the drug secures in-territory regulatory approval",
    "p_reimbursement": "the probability (0-1) the drug is reimbursed on the public formulary",
    "reimbursement_lag_years": "the time in YEARS from approval to public reimbursement listing",
    "regulatory_review_years": "the regulatory review time in YEARS",
    "net_price_usd": "the annual net price per patient in USD",
    "reference_price_factor": "the local net price as a fraction (0-1) of the US/global reference price",
    "peak_share": "the peak market share (0-1) of the branded drug",
    "launch_delay_years": "the launch delay in YEARS beyond regulatory approval",
    "market_access_spend_usd": "the product launch / market-access investment in USD",
}

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "value": {"type": ["number", "null"]},
        "low": {"type": ["number", "null"]},
        "high": {"type": ["number", "null"]},
        "unit": {"type": "string"},
        "quote": {"type": "string"},
        "source_url": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["found", "value", "low", "high", "unit", "quote", "source_url", "confidence"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You extract a single numeric parameter for a biotech territorial-rights valuation "
    "from the SOURCE PASSAGES provided. Rules: (1) Use ONLY the passages — never prior "
    "knowledge or estimation. (2) If the passages do not state the value, set found=false "
    "and value=null. (3) When found, put the exact supporting sentence verbatim in 'quote' "
    "and the URL of the passage it came from in 'source_url'. (4) If the text implies a "
    "range or uncertainty, fill low/high; otherwise set them equal to value. (5) Set "
    "confidence by source authority and directness: 'high' for an authoritative source "
    "stating the figure directly, 'medium' for a credible source needing light "
    "interpretation, 'low' for a weak or indirect source. Never fabricate."
)


def _kind_bounds(key: str) -> str:
    if key in _PROBABILITY_KEYS:
        return "probability"
    if key in _FRACTION_KEYS:
        return "fraction"
    if key in _YEAR_KEYS:
        return "years"
    if key in _USD_KEYS:
        return "usd"
    if key == "epi_rate_per_100k":
        return "per_100k"
    return "fraction"


def _build_user_prompt(
    key: str, results: Sequence[ValyuResult], max_chars: int = _MAX_CHARS_PER_PASSAGE
) -> str:
    ask = _KEY_ASK.get(key, key.replace("_", " "))
    lines = [f"Extract {ask}.", "", "SOURCE PASSAGES:"]
    for i, r in enumerate(results[:_MAX_RESULTS_TO_LLM], 1):
        lines.append(f"\n[{i}] url: {r.url}\n{r.content[:max_chars]}")
    return "\n".join(lines)


class AnthropicClient:
    """Minimal Anthropic Messages client (stdlib urllib, no SDK)."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0):
        if not api_key:
            raise ValueError("Anthropic API key required")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def extract_json(self, system: str, user: str, schema: dict, max_tokens: int = 1024) -> dict:
        body = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }).encode("utf-8")
        req = urllib.request.Request(
            _MESSAGES_URL,
            data=body,
            method="POST",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        # output_config.format guarantees the first text block is valid JSON.
        for block in payload.get("content", []):
            if block.get("type") == "text":
                return json.loads(block["text"])
        raise ValueError("no text block in Anthropic response")


Extractor = Callable[[str, Sequence[ValyuResult]], Optional[SourcedValue]]


def make_llm_extractor(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[AnthropicClient] = None,
    max_chars: int = _MAX_CHARS_PER_PASSAGE,
) -> Extractor:
    """Build an extractor closure with the ``(key, results) -> SourcedValue|None`` shape.

    ``client`` is a seam for tests (inject a fake with ``extract_json``). In
    production, pass nothing: the key comes from ``ANTHROPIC_API_KEY`` and the
    model from ``ANTHROPIC_MODEL`` (default ``claude-opus-4-8``). ``max_chars``
    caps per-passage length — raise it for long DeepResearch reports.
    """
    resolved_model = model or os.getenv("ANTHROPIC_MODEL") or _DEFAULT_MODEL

    def _make_client() -> AnthropicClient:
        return client or AnthropicClient(
            api_key or os.getenv("ANTHROPIC_API_KEY") or "", resolved_model
        )

    def extractor(key: str, results: Sequence[ValyuResult]) -> Optional[SourcedValue]:
        if not results:
            return None
        try:
            data = _make_client().extract_json(
                _SYSTEM, _build_user_prompt(key, results, max_chars), _EXTRACTION_SCHEMA
            )
        except (urllib.error.URLError, TimeoutError, ValueError, OSError, KeyError):
            # A flaky/unauthorized LLM must not sink the valuation — fall back.
            return None
        return build_sourced_value(key, data, resolved_model)

    return extractor


def build_sourced_value(key: str, data: dict, model: str) -> Optional[SourcedValue]:
    """Turn the LLM's structured answer into a SourcedValue, or None (pure)."""
    if not data.get("found") or data.get("value") is None:
        return None
    value = float(data["value"])
    unit = _kind_bounds(key)
    lo_bound, hi_bound = _UNIT_BOUNDS[unit]
    if not (lo_bound <= value <= hi_bound):
        return None  # LLM returned an out-of-range figure — don't trust it

    if key in _PROBABILITY_KEYS:
        kind, low, high = "bernoulli", None, None
    else:
        low, high = data.get("low"), data.get("high")
        if low is not None and high is not None and low < high and low <= value <= high:
            kind = "pert"
        else:
            kind, low, high = "point", None, None

    conf = data.get("confidence")
    if conf not in ("high", "medium", "low"):
        conf = "low"

    return SourcedValue(
        value=value,
        low=low,
        high=high,
        kind=kind,
        unit=unit,
        provenance=Provenance(
            source=(data.get("source_url") or None),
            quote=(data.get("quote") or "")[:400],
            method=f"anthropic {model} extraction (grounded in valyu passages)",
            confidence=conf,
        ),
    )
