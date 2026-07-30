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


# --------------------------------------------------------------------------- #
# BatchDeepResearchProvider — one task per territory, per-field extraction
# --------------------------------------------------------------------------- #
from valuation_engine.research.valyu_deepresearch import BatchDeepResearchProvider  # noqa: E402

_BATCH_REPORT = {"status": "completed",
                 "output": "Net price ~$1200/yr. Reimbursement probability ~45%.",
                 "sources": [{"title": "X", "url": "https://x"}]}


class _BatchClient:
    def __init__(self, task):
        self.task = task
        self.submits = 0

    def submit(self, query, mode=None):
        self.submits += 1
        return "BATCH-TID"

    def poll_until(self, task_id, timeout_s=0, interval_s=0):
        return "completed", self.task


def test_batch_one_submit_across_keys(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    client = _BatchClient(_BATCH_REPORT)

    def ex(key, results):
        assert "Net price" in results[0].content  # each key sees the shared report
        return SourcedValue(value=1.0, kind="point")

    prov = BatchDeepResearchProvider(api_key="k", cache=cache, extractor=ex,
                                     keys=("net_price_usd", "p_reimbursement"))
    prov._make_client = lambda: client
    q1 = ResearchQuery(key="net_price_usd", territory_id="brazil", asset_id="A1", indication="gout")
    q2 = ResearchQuery(key="p_reimbursement", territory_id="brazil", asset_id="A1", indication="gout")
    assert prov.get(q1) is not None and prov.get(q2) is not None
    assert client.submits == 1  # ONE deep task covers both keys
    assert cache.get(prov._report_key(q1))["status"] == "done"
    assert cache.get(cache_key(q1, "deep"))["status"] == "done"


def test_batch_reuses_cached_report(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    q = ResearchQuery(key="net_price_usd", territory_id="brazil", asset_id="A1")
    prov = BatchDeepResearchProvider(api_key="k", cache=cache,
                                     extractor=lambda k, r: SourcedValue(value=2.0, kind="point"))
    cache.set(prov._report_key(q), {"status": "done", "task_id": "T",
                                    "report": "Some report text", "updated_at": None})
    client = _BatchClient(_BATCH_REPORT)
    prov._make_client = lambda: client
    sv = prov.get(q)
    assert sv.value == 2.0 and client.submits == 0  # used cached report, no re-pay


def test_batch_timeout_keeps_report_running(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")

    class _T:
        def submit(self, query, mode=None):
            return "TID"

        def poll_until(self, task_id, timeout_s=0, interval_s=0):
            return "timeout", None

    prov = BatchDeepResearchProvider(api_key="k", cache=cache, extractor=lambda k, r: None)
    prov._make_client = lambda: _T()
    q = ResearchQuery(key="net_price_usd", territory_id="brazil", asset_id="A1")
    assert prov.get(q) is None
    assert cache.get(prov._report_key(q))["status"] == "running"  # resume next run
