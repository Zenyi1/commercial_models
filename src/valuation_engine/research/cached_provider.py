"""Cached evidence provider — versioned JSON, for deterministic/offline runs.

Evidence file shape (all values are ``SourcedValue`` mappings)::

    {
      "assets": {
        "<asset_id>": {"<territory_id>": {"<key>": <SourcedValue>}}
      },
      "territory_defaults": {
        "<territory_id>": {"<key>": <SourcedValue>}
      }
    }

Lookups prefer an asset x territory value, then fall back to a territory
default. This is what lets the whole engine — and its golden test — run without
any live network calls, while keeping every number sourced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.provider import ResearchProvider, ResearchQuery


class CachedProvider(ResearchProvider):
    name = "cached"

    def __init__(self, data: Optional[dict] = None):
        self._assets: dict = (data or {}).get("assets", {})
        self._defaults: dict = (data or {}).get("territory_defaults", {})

    @classmethod
    def from_file(cls, path: str | Path) -> "CachedProvider":
        return cls(json.loads(Path(path).read_text()))

    def available(self) -> bool:
        return bool(self._assets or self._defaults)

    def _lookup(self, table: dict, *keys: str) -> Optional[dict]:
        node = table
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node if isinstance(node, dict) else None

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        raw = None
        if query.asset_id is not None:
            entry = self._lookup(self._assets, query.asset_id, query.territory_id)
            if entry and query.key in entry:
                raw = entry[query.key]
        if raw is None:
            entry = self._defaults.get(query.territory_id)
            if entry and query.key in entry:
                raw = entry[query.key]
        if raw is None:
            return None
        return SourcedValue.model_validate(raw)
