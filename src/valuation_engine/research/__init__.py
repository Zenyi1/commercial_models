"""Evidence layer: turn queries into sourced values.

A :class:`~valuation_engine.research.provider.ResearchProvider` answers a
:class:`~valuation_engine.research.provider.ResearchQuery` (a canonical input
name for an asset x territory) with a ``SourcedValue`` carrying provenance —
never a bare number. Valyu is one backend among several
(:mod:`valuation_engine.research.valyu_provider`,
:mod:`valuation_engine.research.web_provider`, and the structured
:mod:`valuation_engine.research.data_sources` adapters); the
:class:`~valuation_engine.research.provider.MultiSourceProvider` aggregates and
cross-validates them. :class:`~valuation_engine.research.cached_provider.CachedProvider`
serves versioned JSON evidence for deterministic, offline runs.

Guiding rule: LLMs/tools only *find and structure* evidence into
``SourcedValue``s. They never invent point estimates.
"""
