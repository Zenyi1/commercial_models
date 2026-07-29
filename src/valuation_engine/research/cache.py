"""Persistent research cache — pay once per question, never orphan a paid task.

Two problems this solves, both about *paid* research (Valyu DeepResearch bills at
task completion whether or not we collect the result):

1. **Re-paying every run.** A write-through cache keyed by (asset, territory, key)
   stores each *resolved* answer — a value OR a definitive "no figure" — so a
   rerun returns it for free instead of re-submitting.
2. **Orphaning a paid task on timeout.** For async DeepResearch, the cache also
   journals the ``task_id`` and a ``status`` (``running`` / ``done`` / ``failed``).
   If our poll window expires while the task is still running, the entry stays
   ``running`` with its id — the next run *resumes* that task (polling is free;
   only completion is billed, and it already happened) instead of paying again.

Entry shape (one per cache key)::

    {"status": "done"|"running"|"failed", "task_id": <str|null>,
     "answer": <SourcedValue dict|null>, "updated_at": <iso str|null>}

``status == "done"`` is terminal (``answer`` is the value, or null for a
definitive no-figure). ``running`` means a paid task is in flight — resume it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.provider import ResearchProvider, ResearchQuery


def cache_key(q: ResearchQuery) -> str:
    return f"{q.asset_id or '-'}|{q.territory_id}|{q.key}"


class ResearchCache:
    """Thread-safe JSON-backed store with atomic writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self._data = {}

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, entry: dict) -> None:
        with self._lock:
            self._data[key] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)  # atomic

    # -- helpers for storing/reading a resolved answer -------------------- #
    @staticmethod
    def encode_answer(sv: Optional[SourcedValue]) -> Optional[dict]:
        return sv.model_dump(mode="json") if sv is not None else None

    @staticmethod
    def decode_answer(answer: Optional[dict]) -> Optional[SourcedValue]:
        return SourcedValue.model_validate(answer) if answer else None


class CachingProvider(ResearchProvider):
    """Write-through cache over a synchronous provider (e.g. fast Valyu search).

    A cached entry with ``status == "done"`` is returned without calling the
    inner provider — including a cached ``None`` (definitive "no answer"), so we
    don't re-pay to re-learn it. Set ``refresh=True`` to bypass reads and force a
    fresh call (still writes the result back).
    """

    def __init__(self, inner: ResearchProvider, cache: ResearchCache, refresh: bool = False):
        self.inner = inner
        self.cache = cache
        self.refresh = refresh
        self.name = f"cached({inner.name})"

    def available(self) -> bool:
        return self.inner.available()

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        k = cache_key(query)
        if not self.refresh:
            entry = self.cache.get(k)
            if entry is not None and entry.get("status") == "done":
                return ResearchCache.decode_answer(entry.get("answer"))
        sv = self.inner.get(query)
        self.cache.set(k, {"status": "done", "task_id": None,
                           "answer": ResearchCache.encode_answer(sv), "updated_at": None})
        return sv
