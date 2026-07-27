"""Commercialization economics and cost-to-enter.

The intrinsic territory "pie" is operating cash flow = net sales x operating
margin, where margin = 1 - COGS% - SG&A% - distribution%. The *cost to enter*
(regulatory filing, any local bridging trial, and launch market-access spend)
is kept separate from the pie so the go/no-go can compare rNPV of the rights
against the investment required to obtain and exploit them.
"""

from __future__ import annotations

from valuation_engine.core.timeline import present_value_at


def operating_margin(cogs_pct: float, sga_pct: float, distribution_pct: float) -> float:
    """Fraction of net sales that becomes operating cash flow.

    May be negative if assumed cost ratios exceed 100% (a genuine loss signal);
    not clipped, so the model surfaces uneconomic configurations rather than
    hiding them.
    """
    return 1.0 - cogs_pct - sga_pct - distribution_pct


def cost_to_enter_pv(
    filing_cost: float,
    bridging_trial_cost: float,
    market_access_spend: float,
    rate: float,
    market_entry: float,
    convention: str = "mid",
) -> float:
    """Present value of the investment to enter the territory.

    Filing and any bridging trial are treated as spent up front (t=0); launch
    market-access spend is incurred at market entry and discounted back.
    """
    pv = filing_cost + bridging_trial_cost  # spent now (year 0)
    pv += present_value_at(rate, market_access_spend, market_entry, convention)
    return pv
