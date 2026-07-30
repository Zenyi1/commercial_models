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
import time
import urllib.error
import urllib.request
from typing import Optional

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.cache import ResearchCache, cache_key
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


def build_research_question(q: ResearchQuery) -> str:
    """A directive research question for one canonical key × asset × territory."""
    terr = _TERRITORY_NAMES.get(q.territory_id, q.territory_id.replace("_", " "))
    ind = q.indication or "the drug"
    ask = _KEY_ASK.get(q.key, q.key.replace("_", " "))
    return (
        f"For {ind} in {terr}: find {ask}. Report the single most defensible numeric "
        f"value with the exact figure, the unit, and the sources it comes from. If no "
        f"reliable figure exists, say so explicitly rather than estimating."
    )


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


class ValyuDeepResearchProvider(ResearchProvider):
    name = "valyu_deepresearch"

    def __init__(
        self,
        api_key: Optional[str] = None,
        extractor=None,
        mode: str = _DEFAULT_MODE,
        cache: Optional[ResearchCache] = None,
        refresh: bool = False,
        timeout_s: float = _POLL_TIMEOUT_S,
        interval_s: float = _POLL_INTERVAL_S,
        namespace: str = "deep",
    ):
        self.api_key = api_key or os.getenv("VALYU_API_KEY") or os.getenv("VALYU_KEY")
        # Reuse the same grounded LLM extractor; wider char budget for long reports.
        self.extractor = extractor or make_llm_extractor(max_chars=_MAX_REPORT_CHARS)
        self.mode = mode
        self.cache = cache  # doubles as the task journal (task_id + status)
        self.refresh = refresh
        self.timeout_s = timeout_s
        self.interval_s = interval_s
        # Distinct from the fast tier's namespace so a fast-tier cached None does
        # not occupy this key and short-circuit DeepResearch.
        self.namespace = namespace

    def available(self) -> bool:
        return bool(self.api_key)

    def _make_client(self) -> ValyuDeepResearchClient:
        return ValyuDeepResearchClient(self.api_key or "")

    def _journal(self, key: str, status: str, task_id: Optional[str],
                 answer: Optional[dict]) -> None:
        if self.cache is not None:
            self.cache.set(key, {"status": status, "task_id": task_id,
                                 "answer": answer, "updated_at": None})

    def _extract(self, key: str, task: dict) -> Optional[SourcedValue]:
        output = task.get("output") or ""
        if not output:
            return None
        sources = task.get("sources") or []
        src_lines = "\n".join(f"- {s.get('title', '')}: {s.get('url', '')}" for s in sources)
        content = output + ("\n\nSOURCES:\n" + src_lines if src_lines else "")
        result = ValyuResult(
            title="Valyu DeepResearch",
            url=(sources[0].get("url", "") if sources else ""),
            content=content, source="valyu deepresearch", relevance=0.9,
        )
        return self.extractor(key, [result])

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        if not self.available():
            return None
        k = cache_key(query, self.namespace)
        entry = self.cache.get(k) if self.cache is not None else None

        # Resolved before -> return it (value or definitive None). No re-pay.
        if not self.refresh and entry is not None and entry.get("status") == "done":
            return ResearchCache.decode_answer(entry.get("answer"))

        client = self._make_client()
        # Resume a journaled, still-running paid task instead of submitting anew.
        task_id = entry.get("task_id") if (entry and entry.get("status") == "running") else None
        try:
            if task_id is None:
                task_id = client.submit(build_research_question(query), mode=self.mode)
                self._journal(k, "running", task_id, None)
            outcome, task = client.poll_until(task_id, self.timeout_s, self.interval_s)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError, KeyError):
            return None

        if outcome == "timeout":
            self._journal(k, "running", task_id, None)  # keep id; resume next run
            return None
        if outcome != "completed" or not task:
            self._journal(k, "failed", task_id, None)
            return None

        sv = self._extract(query.key, task)
        self._journal(k, "done", task_id, ResearchCache.encode_answer(sv))
        return sv
