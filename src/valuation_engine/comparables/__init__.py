"""Comparable-deal data foundation.

The largest attainable universe of historical biotech deals (any geography) is
normalized into :class:`~valuation_engine.comparables.schema.DealRecord`s
(``harvest``), then regressed by ``analyzer`` into benchmark royalty/upfront
ranges and the **geography-adjustment factors** that project global deal
evidence onto specific emerging markets. ``triangulation`` uses those outputs to
cross-check the bottom-up rNPV.
"""
