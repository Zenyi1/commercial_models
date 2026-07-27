"""Valyu deep-research backend (key-gated).

This is the seam for Valyu. It is *available* only when ``VALYU_API_KEY`` is set
and a client can be constructed; otherwise ``get`` returns None and the engine
falls back to other providers / cached evidence. The actual retrieval+extraction
call is intentionally left as a single clearly-marked method so wiring Valyu is a
localized change — and, critically, the extracted result must be returned as a
``SourcedValue`` with a real source URL and quote (never an unsourced number).
"""

from __future__ import annotations

import os
from typing import Optional

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.provider import ResearchProvider, ResearchQuery


class ValyuProvider(ResearchProvider):
    name = "valyu"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("VALYU_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        if not self.available():
            return None
        return self._research(query)

    # ------------------------------------------------------------------ #
    # Wire Valyu here. Must return a SourcedValue with provenance.source set
    # to the retrieved citation and provenance.quote to the supporting text,
    # or None if no confident answer is found. Returning None is correct and
    # safe — the aggregator will try other sources. Do NOT fabricate a value.
    # ------------------------------------------------------------------ #
    def _research(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        raise NotImplementedError(
            "ValyuProvider._research is a wiring point: call Valyu, extract a "
            "sourced value for query.key, and return a SourcedValue with provenance."
        )
