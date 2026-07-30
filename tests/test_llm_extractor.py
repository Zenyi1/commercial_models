"""Tests for the LLM extraction layer — grounded reader over Valyu passages.

All keyless/networkless: the Anthropic client is replaced with a fake that
returns a canned structured-output dict, so we exercise the prompt-building,
SourcedValue construction, grounding (found=false -> None), and range-guard
logic without a real API call."""

import pytest

from valuation_engine.research.valyu_provider import ValyuResult
from valuation_engine.research.llm_extractor import (
    AnthropicClient,
    build_sourced_value,
    make_llm_extractor,
)


class _FakeAnthropic:
    """Stand-in for AnthropicClient — returns a fixed structured answer."""

    def __init__(self, answer):
        self.answer = answer
        self.last_user = None

    def extract_json(self, system, user, schema, max_tokens=1024):
        self.last_user = user
        return self.answer


RESULTS = [
    ValyuResult(
        title="CONITEC report",
        url="https://gov.br/conitec",
        content="Only about 45% of branded gout drugs are recommended for public reimbursement.",
        source="gov.br",
        relevance=0.9,
    )
]


def test_llm_extractor_builds_sourced_value():
    fake = _FakeAnthropic({
        "found": True, "value": 0.45, "low": 0.35, "high": 0.55,
        "unit": "probability", "quote": "Only about 45% ...",
        "source_url": "https://gov.br/conitec", "confidence": "high",
    })
    extract = make_llm_extractor(client=fake, model="claude-haiku-4-5")
    sv = extract("p_reimbursement", RESULTS)
    assert sv is not None
    assert sv.value == pytest.approx(0.45)
    assert sv.kind == "bernoulli"  # probability keys are bernoulli, low/high dropped
    assert sv.low is None and sv.high is None
    assert sv.provenance.source == "https://gov.br/conitec"
    assert sv.provenance.confidence == "high"
    assert "anthropic" in sv.provenance.method and "haiku" in sv.provenance.method


def test_llm_extractor_passes_passages_in_prompt():
    fake = _FakeAnthropic({"found": False, "value": None, "low": None, "high": None,
                           "unit": "", "quote": "", "source_url": "", "confidence": "low"})
    extract = make_llm_extractor(client=fake)
    extract("p_reimbursement", RESULTS)
    assert "https://gov.br/conitec" in fake.last_user
    assert "45%" in fake.last_user


def test_llm_extractor_found_false_returns_none():
    fake = _FakeAnthropic({"found": False, "value": None, "low": None, "high": None,
                           "unit": "", "quote": "", "source_url": "", "confidence": "low"})
    extract = make_llm_extractor(client=fake)
    assert extract("net_price_usd", RESULTS) is None


def test_llm_extractor_range_becomes_pert():
    # A non-probability fraction with a valid range -> pert with low/high kept.
    sv = build_sourced_value(
        "treatment_rate",
        {"found": True, "value": 0.4, "low": 0.3, "high": 0.5, "unit": "fraction",
         "quote": "~40% treated", "source_url": "u", "confidence": "medium"},
        "claude-opus-4-8",
    )
    assert sv is not None and sv.kind == "pert"
    assert sv.low == pytest.approx(0.3) and sv.high == pytest.approx(0.5)


def test_llm_extractor_rejects_out_of_range_probability():
    # LLM hallucinated 1.4 for a probability -> dropped, not emitted.
    sv = build_sourced_value(
        "p_reimbursement",
        {"found": True, "value": 1.4, "low": None, "high": None, "unit": "probability",
         "quote": "q", "source_url": "u", "confidence": "high"},
        "claude-opus-4-8",
    )
    assert sv is None


def test_llm_extractor_swallows_backend_errors():
    class _Boom:
        def extract_json(self, *a, **k):
            raise OSError("anthropic down")

    extract = make_llm_extractor(client=_Boom())
    assert extract("p_reimbursement", RESULTS) is None


def test_anthropic_client_requires_key():
    with pytest.raises(ValueError):
        AnthropicClient(api_key="", model="claude-opus-4-8")


def test_estimate_mode_uses_estimate_prompt_and_low_confidence():
    captured = {}

    class _F:
        def extract_json(self, system, user, schema, max_tokens=1024):
            captured["system"] = system
            return {"found": True, "value": 0.4, "low": 0.3, "high": 0.5, "unit": "fraction",
                    "quote": "derived from stated counts", "source_url": "u", "confidence": "low"}

    ex = make_llm_extractor(client=_F(), allow_estimate=True)
    sv = ex("treatment_rate", RESULTS)
    assert sv is not None and sv.kind == "pert" and sv.provenance.confidence == "low"
    assert "estimate" in captured["system"].lower()  # used the estimate system prompt
