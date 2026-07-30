"""Tests for the EscalatingProvider ladder and BatchDeepResearchProvider
(one task per territory, per-field extraction, cache/journal). No network."""

import pytest

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.cache import ResearchCache, cache_key
from valuation_engine.research.provider import (
    EscalatingProvider,
    ResearchProvider,
    ResearchQuery,
)
from valuation_engine.research.valyu_deepresearch import (
    BatchDeepResearchProvider,
    build_combined_question,
)


# --------------------------------------------------------------------------- #
# EscalatingProvider — try in order, first non-None
# --------------------------------------------------------------------------- #
class _Stub(ResearchProvider):
    def __init__(self, name, answer, avail=True):
        self.name, self.answer, self._avail, self.calls = name, answer, avail, 0

    def available(self):
        return self._avail

    def get(self, query):
        self.calls += 1
        return self.answer


def _sv(v=0.5):
    return SourcedValue(value=v, kind="bernoulli")


def _q(key="net_price_usd"):
    return ResearchQuery(key=key, territory_id="brazil", asset_id="A1", indication="gout")


def test_escalate_uses_second_when_first_returns_none():
    fast, deep = _Stub("fast", None), _Stub("deep", _sv(0.42))
    sv = EscalatingProvider([fast, deep]).get(_q())
    assert sv is not None and sv.value == pytest.approx(0.42)
    assert fast.calls == 1 and deep.calls == 1


def test_escalate_skips_second_when_first_answers():
    fast, deep = _Stub("fast", _sv(0.9)), _Stub("deep", _sv(0.1))
    sv = EscalatingProvider([fast, deep]).get(_q("p_reimbursement"))
    assert sv.value == pytest.approx(0.9)
    assert deep.calls == 0  # deep tier never touched — no wasted cost


def test_escalate_all_none_returns_none():
    prov = EscalatingProvider([_Stub("a", None), _Stub("b", None)])
    assert prov.get(_q("peak_share")) is None


def test_build_combined_question_covers_all_keys():
    q = build_combined_question(_q(), ("net_price_usd", "p_reimbursement"))
    assert "gout" in q and "Brazil" in q and "EACH" in q


# --------------------------------------------------------------------------- #
# BatchDeepResearchProvider — one task per territory, per-field extraction
# --------------------------------------------------------------------------- #
_BATCH_REPORT = {"status": "completed",
                 "output": "Net price ~$1200/yr. Reimbursement probability ~45%.",
                 "sources": [{"title": "X", "url": "https://x"}]}


class _BatchClient:
    def __init__(self, task):
        self.task, self.submits = task, 0

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
    assert prov.get(_q("net_price_usd")) is not None
    assert prov.get(_q("p_reimbursement")) is not None
    assert client.submits == 1  # ONE deep task covers both keys
    assert cache.get(prov._report_key(_q()))["status"] == "done"
    assert cache.get(cache_key(_q(), "deep"))["status"] == "done"


def test_batch_reuses_cached_report(tmp_path):
    cache = ResearchCache(tmp_path / "c.json")
    prov = BatchDeepResearchProvider(api_key="k", cache=cache,
                                     extractor=lambda k, r: SourcedValue(value=2.0, kind="point"))
    cache.set(prov._report_key(_q()), {"status": "done", "task_id": "T",
                                       "report": "Some report text", "updated_at": None})
    client = _BatchClient(_BATCH_REPORT)
    prov._make_client = lambda: client
    sv = prov.get(_q())
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
    assert prov.get(_q()) is None
    assert cache.get(prov._report_key(_q()))["status"] == "running"  # resume next run


def test_batch_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("VALYU_API_KEY", raising=False)
    monkeypatch.delenv("VALYU_KEY", raising=False)
    prov = BatchDeepResearchProvider(api_key=None, extractor=lambda k, r: None)
    assert prov.available() is False
    assert prov.get(_q()) is None
