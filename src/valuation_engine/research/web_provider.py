"""Generic web-search + targeted-crawl backend (config-gated).

Available only when a search/crawl backend is configured (``WEB_SEARCH_API_KEY``).
Intended for regulatory-agency pages and deal press releases that structured
feeds don't cover. Like every backend it must return a ``SourcedValue`` with a
real source URL, or None.
"""

from __future__ import annotations

import os
from typing import Optional

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.provider import ResearchProvider, ResearchQuery


class WebProvider(ResearchProvider):
    name = "web"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("WEB_SEARCH_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        if not self.available():
            return None
        return self._search(query)

    def _search(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        raise NotImplementedError(
            "WebProvider._search is a wiring point: search/crawl, extract a sourced "
            "value for query.key, and return a SourcedValue with provenance."
        )
