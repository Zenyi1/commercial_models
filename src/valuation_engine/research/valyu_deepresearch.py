"""Valyu DeepResearch backend — the slow/expensive escalation tier.

Fast :mod:`valyu_provider` search answers most keys in seconds. For the residual
it returns ``None`` on — figures no single passage states, but a multi-step
research agent can synthesize (e.g. the net price of a branded drug in a market) —
this provider runs Valyu DeepResearch: an async task that researches across many
sources and returns a cited report, which the LLM extractor then reads for the
number. Compose behind :class:`~valuation_engine.research.provider.EscalatingProvider`
so it fires only on keys fast search couldn't answer.

Async by nature: submit a task, poll to completion (minutes), then extract. The
poll is bounded (``_POLL_TIMEOUT_S``) so a stuck task degrades to ``None`` rather
than blocking forever. Stdlib urllib only — no SDK.

--------------------------------------------------------------------------------
API surface (docs.valyu.ai, 2026-07): submit ``POST
https://api.valyu.ai/v1/deepresearch/tasks`` (x-api-key) -> ``deepresearch_id``;
poll ``GET /v1/deepresearch/tasks/{id}/status`` -> ``status`` (queued/running/
completed/failed/cancelled/…) and, when completed, ``output`` (report markdown) +
``sources`` [{title,url}]. Endpoint/fields isolated as constants below.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.cache import ResearchCache, cache_key
from valuation_engine.research.enrich import _OVERRIDE_FIELDS
from valuation_engine.research.llm_extractor import _KEY_ASK, make_llm_extractor
from valuation_engine.research.provider import ResearchProvider, ResearchQuery
from valuation_engine.research.valyu_provider import ValyuResult, _TERRITORY_NAMES

# -- Valyu DeepResearch API surface (verify against docs; isolated here) ------- #
_TASKS_URL = "https://api.valyu.ai/v1/deepresearch/tasks"
_AUTH_HEADER = "x-api-key"
_DEFAULT_MODE = os.getenv("VALYU_DEEPRESEARCH_MODE", "fast")  # fast|standard|heavy|max
_POLL_INTERVAL_S = 8.0
# A single fast task takes ~200s to complete; under concurrent load it's longer,
# and `output`/`sources` only appear on /status once status == "completed". Give
# it real headroom so tasks aren't abandoned mid-flight (override if needed).
_POLL_TIMEOUT_S = float(os.getenv("VALYU_DEEPRESEARCH_TIMEOUT_S", "600"))
_TERMINAL_OK = "completed"
_TERMINAL_FAIL = {"failed", "cancelled"}
_MAX_REPORT_CHARS = 24000  # reports run ~30k chars; send most of it to the extractor


def build_combined_question(query: ResearchQuery, keys: Sequence[str]) -> str:
    """One research question covering every residual key for an asset × territory."""
    terr = _TERRITORY_NAMES.get(query.territory_id, query.territory_id.replace("_", " "))
    ind = query.indication or "the drug"
    items = "\n".join(f"- {_KEY_ASK.get(k, k.replace('_', ' '))}" for k in keys)
    return (
        f"For {ind} in {terr}, research and report specific numeric values for EACH of "
        f"the following, with the figure, its unit, and the source for each:\n{items}\n"
        f"For any item without a reliable figure, say so explicitly for that item."
    )


def _report_text(task: dict) -> str:
    """Flatten a completed DeepResearch task into report text + a sources list."""
    output = task.get("output") or ""
    if not output:
        return ""
    sources = task.get("sources") or []
    src_lines = "\n".join(f"- {s.get('title', '')}: {s.get('url', '')}" for s in sources)
    return output + ("\n\nSOURCES:\n" + src_lines if src_lines else "")


class ValyuDeepResearchClient:
    """Minimal Valyu DeepResearch client (submit + poll). Stdlib urllib, no SDK."""

    def __init__(self, api_key: str, timeout_s: float = 30.0):
        if not api_key:
            raise ValueError("Valyu API key required")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict:
        return {_AUTH_HEADER: self.api_key, "Content-Type": "application/json",
                "Accept": "application/json"}

    def submit(self, query: str, mode: str = _DEFAULT_MODE) -> str:
        body = json.dumps({"query": query, "mode": mode,
                           "output_formats": ["markdown"]}).encode("utf-8")
        req = urllib.request.Request(_TASKS_URL, data=body, method="POST",
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        task_id = payload.get("deepresearch_id") or payload.get("id")
        if not task_id:
            raise ValueError("no deepresearch_id in submit response")
        return str(task_id)

    def poll(self, task_id: str) -> dict:
        req = urllib.request.Request(f"{_TASKS_URL}/{task_id}/status", method="GET",
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def poll_until(
        self, task_id: str, timeout_s: float = _POLL_TIMEOUT_S,
        interval_s: float = _POLL_INTERVAL_S,
    ) -> tuple[str, Optional[dict]]:
        """Poll an existing task. Returns (outcome, task):
        ("completed", task) | ("failed", None) | ("timeout", None).

        Polling an in-flight task is free — only completion is billed, and by the
        time we resume a journaled id it has usually already completed — so this
        collects work we already paid for instead of re-submitting.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            task = self.poll(task_id)
            status = task.get("status")
            if status == _TERMINAL_OK:
                return "completed", task
            if status in _TERMINAL_FAIL:
                return "failed", None
            time.sleep(interval_s)
        return "timeout", None


class BatchDeepResearchProvider(ResearchProvider):
    """One DeepResearch task per (asset, territory) covering ALL residual keys,
    with a per-field LLM extraction off the shared cached report.

    Cheaper and faster than a task per key: ``1 deep task + N cheap extractions``
    instead of ``N deep tasks``. The report is cached and journaled per territory
    (so a timeout resumes the single task, and re-extraction — e.g. after turning
    on estimate mode — is free). Each field's answer is cached under its own key.
    A per-territory lock ensures the task is submitted once even under the
    parallel enrich fan-out.
    """

    name = "valyu_deepresearch_batch"

    def __init__(
        self,
        api_key: Optional[str] = None,
        keys: Optional[Sequence[str]] = None,
        extractor=None,
        mode: str = _DEFAULT_MODE,
        cache: Optional[ResearchCache] = None,
        refresh: bool = False,
        timeout_s: float = _POLL_TIMEOUT_S,
        interval_s: float = _POLL_INTERVAL_S,
        answer_namespace: str = "deep",
        report_namespace: str = "deep_report",
    ):
        self.api_key = api_key or os.getenv("VALYU_API_KEY") or os.getenv("VALYU_KEY")
        self.keys = tuple(keys) if keys else tuple(_OVERRIDE_FIELDS)
        self.extractor = extractor or make_llm_extractor(max_chars=_MAX_REPORT_CHARS)
        self.mode = mode
        self.cache = cache
        self.refresh = refresh
        self.timeout_s = timeout_s
        self.interval_s = interval_s
        self.answer_namespace = answer_namespace
        self.report_namespace = report_namespace
        self._locks: dict = {}
        self._locks_guard = threading.Lock()

    def available(self) -> bool:
        return bool(self.api_key)

    def _make_client(self) -> ValyuDeepResearchClient:
        return ValyuDeepResearchClient(self.api_key or "")

    def _lock_for(self, rk: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(rk, threading.Lock())

    def _report_key(self, q: ResearchQuery) -> str:
        return f"{self.report_namespace}|{q.asset_id or '-'}|{q.territory_id}"

    def _set_report(self, rk: str, status: str, task_id: Optional[str], report: Optional[str]) -> None:
        if self.cache is not None:
            self.cache.set(rk, {"status": status, "task_id": task_id,
                                "report": report, "updated_at": None})

    def _ensure_report(self, query: ResearchQuery) -> Optional[str]:
        """Submit (once) and return the combined report for this territory, or None."""
        rk = self._report_key(query)
        if self.cache is not None and not self.refresh:
            e = self.cache.get(rk)
            if e is not None and e.get("status") == "done":
                return e.get("report")
        with self._lock_for(rk):
            e = self.cache.get(rk) if self.cache is not None else None
            if e is not None and e.get("status") == "done" and not self.refresh:
                return e.get("report")  # another thread completed it
            client = self._make_client()
            task_id = e.get("task_id") if (e and e.get("status") == "running") else None
            if task_id is None:
                task_id = client.submit(build_combined_question(query, self.keys), mode=self.mode)
                self._set_report(rk, "running", task_id, None)
            outcome, task = client.poll_until(task_id, self.timeout_s, self.interval_s)
            if outcome == "timeout":
                self._set_report(rk, "running", task_id, None)  # resume next run
                return None
            if outcome != "completed" or not task:
                self._set_report(rk, "failed", task_id, None)
                return None
            report = _report_text(task)
            self._set_report(rk, "done", task_id, report or None)
            return report or None

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        if not self.available():
            return None
        ak = cache_key(query, self.answer_namespace)
        if self.cache is not None and not self.refresh:
            e = self.cache.get(ak)
            if e is not None and e.get("status") == "done":
                return ResearchCache.decode_answer(e.get("answer"))
        try:
            report = self._ensure_report(query)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError, KeyError):
            return None
        if not report:
            return None
        result = ValyuResult(title="Valyu DeepResearch (batch)", url="",
                             content=report, source="valyu deepresearch", relevance=0.9)
        sv = self.extractor(query.key, [result])
        if self.cache is not None:
            self.cache.set(ak, {"status": "done", "task_id": None,
                                "answer": ResearchCache.encode_answer(sv), "updated_at": None})
        return sv
