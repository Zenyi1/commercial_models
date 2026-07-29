"""Tests for the DeepResearch escalation tier and the EscalatingProvider ladder.
No network — fake clients/providers stand in."""

import pytest

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.provider import (
    EscalatingProvider,
    ResearchProvider,
    ResearchQuery,
)
from valuation_engine.research.valyu_deepresearch import (
    ValyuDeepResearchProvider,
    build_research_question,
)


# --------------------------------------------------------------------------- #
# EscalatingProvider
# --------------------------------------------------------------------------- #
class _Stub(ResearchProvider):
    def __init__(self, name, answer, avail=True):
        self.name = name
        self.answer = answer
        self._avail = avail
        self.calls = 0

    def available(self):
        return self._avail

    def get(self, query):
        self.calls += 1
        return self.answer


def _sv(v=0.5):
    return SourcedValue(value=v, kind="bernoulli")


def test_escalate_uses_second_when_first_returns_none():
    fast = _Stub("fast", None)
    deep = _Stub("deep", _sv(0.42))
    prov = EscalatingProvider([fast, deep])
    sv = prov.get(ResearchQuery(key="net_price_usd", territory_id="brazil"))
    assert sv is not None and sv.value == pytest.approx(0.42)
    assert fast.calls == 1 and deep.calls == 1  # escalated


def test_escalate_skips_second_when_first_answers():
    fast = _Stub("fast", _sv(0.9))
    deep = _Stub("deep", _sv(0.1))
    prov = EscalatingProvider([fast, deep])
    sv = prov.get(ResearchQuery(key="p_reimbursement", territory_id="brazil"))
    assert sv.value == pytest.approx(0.9)
    assert deep.calls == 0  # deep tier never touched — no wasted cost


def test_escalate_all_none_returns_none():
    prov = EscalatingProvider([_Stub("a", None), _Stub("b", None)])
    assert prov.get(ResearchQuery(key="peak_share", territory_id="brazil")) is None


def test_escalate_skips_unavailable_provider():
    fast = _Stub("fast", _sv(0.3), avail=False)
    deep = _Stub("deep", _sv(0.7))
    prov = EscalatingProvider([fast, deep])
    assert prov.get(ResearchQuery(key="net_price_usd", territory_id="brazil")).value == pytest.approx(0.7)
    assert fast.calls == 0  # unavailable -> not called


# --------------------------------------------------------------------------- #
# ValyuDeepResearchProvider (fake client + fake extractor)
# --------------------------------------------------------------------------- #
def test_build_research_question_is_directive():
    q = build_research_question(ResearchQuery(key="net_price_usd", territory_id="mexico", indication="gout"))
    assert "gout" in q and "Mexico" in q and "reliable figure" in q


def test_deepresearch_provider_extracts_from_report():
    class _FakeClient:
        def research(self, query, **kw):
            return {"status": "completed", "output": "Net price is about $1,200/yr.",
                    "sources": [{"title": "PriceRef", "url": "https://ref"}]}

    captured = {}

    def fake_extractor(key, results):
        captured["content"] = results[0].content
        captured["url"] = results[0].url
        return SourcedValue(value=1200.0, kind="point", unit="usd")

    prov = ValyuDeepResearchProvider(api_key="k", extractor=fake_extractor)
    prov._make_client = lambda: _FakeClient()
    sv = prov.get(ResearchQuery(key="net_price_usd", territory_id="mexico", indication="gout"))
    assert sv is not None and sv.value == pytest.approx(1200.0)
    assert "1,200" in captured["content"]
    assert "SOURCES:" in captured["content"] and "https://ref" in captured["content"]
    assert captured["url"] == "https://ref"


def test_deepresearch_none_when_not_completed():
    class _FakeClient:
        def research(self, query, **kw):
            return None  # timed out / failed

    prov = ValyuDeepResearchProvider(api_key="k", extractor=lambda k, r: None)
    prov._make_client = lambda: _FakeClient()
    assert prov.get(ResearchQuery(key="net_price_usd", territory_id="mexico")) is None


def test_deepresearch_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("VALYU_API_KEY", raising=False)
    monkeypatch.delenv("VALYU_KEY", raising=False)
    prov = ValyuDeepResearchProvider(api_key=None, extractor=lambda k, r: None)
    assert prov.available() is False
    assert prov.get(ResearchQuery(key="net_price_usd", territory_id="mexico")) is None


def test_deepresearch_swallows_backend_errors():
    class _Boom:
        def research(self, query, **kw):
            raise OSError("valyu down")

    prov = ValyuDeepResearchProvider(api_key="k", extractor=lambda k, r: None)
    prov._make_client = lambda: _Boom()
    assert prov.get(ResearchQuery(key="net_price_usd", territory_id="mexico")) is None
