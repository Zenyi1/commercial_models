"""Tests for the DeepResearch escalation tier, the EscalatingProvider ladder,
and the cache/task-journal (resume-not-re-pay). No network — fakes stand in."""

import pytest

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.cache import ResearchCache, cache_key
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
    fast, deep = _Stub("fast", None), _Stub("deep", _sv(0.42))
    sv = EscalatingProvider([fast, deep]).get(ResearchQuery(key="net_price_usd", territory_id="brazil"))
    assert sv is not None and sv.value == pytest.approx(0.42)
    assert fast.calls == 1 and deep.calls == 1


def test_escalate_skips_second_when_first_answers():
    fast, deep = _Stub("fast", _sv(0.9)), _Stub("deep", _sv(0.1))
    sv = EscalatingProvider([fast, deep]).get(ResearchQuery(key="p_reimbursement", territory_id="brazil"))
    assert sv.value == pytest.approx(0.9)
    assert deep.calls == 0  # deep tier never touched — no wasted cost


def test_escalate_all_none_returns_none():
    prov = EscalatingProvider([_Stub("a", None), _Stub("b", None)])
    assert prov.get(ResearchQuery(key="peak_share", territory_id="brazil")) is None


def test_build_research_question_is_directive():
    q = build_research_question(ResearchQuery(key="net_price_usd", territory_id="mexico", indication="gout"))
    assert "gout" in q and "Mexico" in q and "reliable figure" in q


# --------------------------------------------------------------------------- #
# DeepResearch provider — client uses submit() + poll_until()
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self, outcome="completed", task=None):
        self.outcome, self.task = outcome, task
        self.submits, self.polls, self.last_id = 0, 0, None

    def submit(self, query, mode=None):
        self.submits += 1
        return "TID-new"

    def poll_until(self, task_id, timeout_s=0, interval_s=0):
        self.polls += 1
        self.last_id = task_id
        return self.outcome, self.task


_COMPLETED = {"status": "completed", "output": "Net price is about $1,200/yr.",
              "sources": [{"title": "PriceRef", "url": "https://ref"}]}


def _provider(client, cache=None, extractor=None):
    prov = ValyuDeepResearchProvider(api_key="k", cache=cache,
                                     extractor=extractor or (lambda k, r: SourcedValue(value=1200.0, kind="point", unit="usd")))
    prov._make_client = lambda: client
    return prov


def test_deepresearch_extracts_and_caches(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    client = _FakeClient(outcome="completed", task=_COMPLETED)
    q = ResearchQuery(key="net_price_usd", territory_id="mexico", asset_id="A1", indication="gout")
    sv = _provider(client, cache).get(q)
    assert sv is not None and sv.value == pytest.approx(1200.0)
    assert client.submits == 1
    entry = cache.get(cache_key(q, "deep"))
    assert entry["status"] == "done" and entry["answer"]["value"] == pytest.approx(1200.0)


def test_deepresearch_resolved_cache_hit_skips_client(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    q = ResearchQuery(key="net_price_usd", territory_id="mexico", asset_id="A1")
    cache.set(cache_key(q, "deep"), {"status": "done", "task_id": "old",
                             "answer": {"value": 999.0, "kind": "point"}, "updated_at": None})
    client = _FakeClient()
    sv = _provider(client, cache).get(q)
    assert sv.value == pytest.approx(999.0)
    assert client.submits == 0 and client.polls == 0  # never touched the paid API


def test_deepresearch_resumes_running_task_instead_of_resubmitting(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    q = ResearchQuery(key="net_price_usd", territory_id="mexico", asset_id="A1")
    cache.set(cache_key(q, "deep"), {"status": "running", "task_id": "TID-old", "answer": None, "updated_at": None})
    client = _FakeClient(outcome="completed", task=_COMPLETED)
    sv = _provider(client, cache).get(q)
    assert sv is not None
    assert client.submits == 0          # did NOT re-submit (no re-pay)
    assert client.last_id == "TID-old"  # resumed the journaled paid task
    assert cache.get(cache_key(q, "deep"))["status"] == "done"


def test_deepresearch_timeout_keeps_task_for_resume(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    q = ResearchQuery(key="net_price_usd", territory_id="mexico", asset_id="A1")
    client = _FakeClient(outcome="timeout", task=None)
    sv = _provider(client, cache).get(q)
    assert sv is None
    entry = cache.get(cache_key(q, "deep"))
    assert entry["status"] == "running" and entry["task_id"] == "TID-new"  # not orphaned


def test_deepresearch_completed_but_no_figure_caches_definitive_none(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    q = ResearchQuery(key="net_price_usd", territory_id="mexico", asset_id="A1")
    client = _FakeClient(outcome="completed", task=_COMPLETED)
    prov = _provider(client, cache, extractor=lambda k, r: None)  # LLM found nothing
    assert prov.get(q) is None
    entry = cache.get(cache_key(q, "deep"))
    assert entry["status"] == "done" and entry["answer"] is None  # won't re-pay


def test_deepresearch_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("VALYU_API_KEY", raising=False)
    monkeypatch.delenv("VALYU_KEY", raising=False)
    prov = ValyuDeepResearchProvider(api_key=None, extractor=lambda k, r: None)
    assert prov.available() is False
    assert prov.get(ResearchQuery(key="net_price_usd", territory_id="mexico")) is None


def test_deepresearch_swallows_submit_errors(tmp_path):
    class _Boom:
        def submit(self, *a, **k):
            raise OSError("valyu down")

    prov = _provider(_Boom(), ResearchCache(tmp_path / "c.json"), extractor=lambda k, r: None)
    assert prov.get(ResearchQuery(key="net_price_usd", territory_id="mexico", asset_id="A1")) is None
