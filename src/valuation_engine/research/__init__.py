"""Evidence layer: turn queries into sourced values.

A :class:`~valuation_engine.research.provider.ResearchProvider` answers a
:class:`~valuation_engine.research.provider.ResearchQuery` (a canonical input
name for an asset x territory) with a ``SourcedValue`` carrying provenance —
never a bare number. Backends: :mod:`valuation_engine.research.valyu_provider`
(fast search + LLM extraction) and :mod:`valuation_engine.research.valyu_deepresearch`
(the escalation tier), composed via
:class:`~valuation_engine.research.provider.EscalatingProvider` /
:class:`~valuation_engine.research.provider.MultiSourceProvider`.
:class:`~valuation_engine.research.cache.CachingProvider` makes paid research
pay-once; :class:`~valuation_engine.research.cached_provider.CachedProvider`
serves hand-authored versioned JSON evidence for deterministic, offline runs.

Guiding rule: LLMs/tools only *find and structure* evidence into
``SourcedValue``s. They never invent point estimates.
"""
