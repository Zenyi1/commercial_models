"""Structured, sourced data adapters.

Currently ships :mod:`valuation_engine.research.data_sources.edgar` (SEC EDGAR
deals harvest). Authoritative-feed adapters (WHO/IHME, HTA, pricing) are not
implemented — wire them here as :class:`ResearchProvider`s when a feed is added.
"""
