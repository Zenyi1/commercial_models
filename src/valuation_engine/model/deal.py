"""Deal-split layer.

Given the intrinsic territory pie (rNPV of operating cash flow) plus the annual
net-sales stream, allocate value between the parties under a deal structure:

* **Licensor** receives upfront + milestones + royalties (PV).
* **Licensee** commercializes, earning the pie, and pays those amounts:
  ``licensee_value = pie - upfront - milestones - royalties``.

By construction ``licensor_value + licensee_value == pie`` — the deal is a
zero-sum split of the same cash flows (cost-base/synergy differences between
parties are out of scope). Milestones are risk-weighted by the gate they depend
on (approval, reimbursement) using the *same* probability parameters as the
revenue model, so base case and Monte Carlo stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from valuation_engine.core.timeline import npv, present_value_at
from valuation_engine.inputs.schemas import Milestone, RoyaltyTier


def royalty_by_year(
    net_sales: np.ndarray, tiers: Sequence[RoyaltyTier]
) -> np.ndarray:
    """Marginal tiered royalty applied to each year's net sales.

    Each tier taxes the band of annual net sales in ``[min, max)`` at its rate.
    With no tiers, royalties are zero.
    """
    if not tiers:
        return np.zeros_like(net_sales, dtype=float)
    royalty = np.zeros_like(net_sales, dtype=float)
    for tier in tiers:
        lo = tier.min_sales_usd
        hi = tier.max_sales_usd if tier.max_sales_usd is not None else np.inf
        band = np.clip(net_sales, lo, hi) - lo
        band = np.maximum(band, 0.0)
        royalty = royalty + band * tier.rate
    return royalty


@dataclass
class MilestoneFlow:
    label: str
    year: float
    expected_amount: float
    pv: float


@dataclass
class DealValuation:
    pie_rnpv: float
    upfront_pv: float
    milestone_pv: float
    royalty_pv: float
    licensor_value: float
    licensee_value: float
    milestone_flows: list[MilestoneFlow] = field(default_factory=list)


def _milestone_timing_and_weight(
    m: Milestone,
    params: Mapping[str, float],
    idx: int,
    market_entry: float,
    public_start: float,
    commercialization_weight: float,
    p_reimbursement: float,
    cumulative_sales: np.ndarray,
    years: np.ndarray,
) -> tuple[float, float] | None:
    """Return ``(year, weight)`` for a milestone, or None if it is never triggered."""
    if m.probability_override is not None:
        weight = m.probability_override.base()
    else:
        weight = commercialization_weight

    if m.trigger == "on_approval":
        return market_entry, weight
    if m.trigger == "on_reimbursement":
        return public_start, weight * p_reimbursement
    if m.trigger == "at_year":
        year = m.year_from_now.base() if m.year_from_now is not None else 0.0
        return year, weight
    if m.trigger == "on_sales":
        if m.sales_threshold_usd is None:
            return None
        threshold = m.sales_threshold_usd.base()
        crossed = np.where(cumulative_sales >= threshold)[0]
        if crossed.size == 0:
            return None
        # Paid mid-year of the crossing period. Sales already carry the launch
        # probability weight, so no extra weighting here.
        year = float(years[crossed[0]]) + 0.5
        w = 1.0 if m.probability_override is None else weight
        return year, w
    return None


def value_deal(
    pie_rnpv: float,
    net_sales: np.ndarray,
    years: np.ndarray,
    rate: float,
    params: Mapping[str, float],
    milestones: Sequence[Milestone],
    royalty_tiers: Sequence[RoyaltyTier],
    market_entry: float,
    public_start: float,
    commercialization_weight: float,
    p_reimbursement: float,
    convention: str = "mid",
) -> DealValuation:
    upfront = float(params.get("upfront_usd", 0.0))
    # Upfront is paid at signing (t=0) regardless of outcome — undiscounted.
    upfront_pv = upfront

    royalty = royalty_by_year(net_sales, royalty_tiers)
    royalty_pv = npv(rate, royalty, convention)

    cumulative_sales = np.cumsum(net_sales)
    milestone_pv = 0.0
    flows: list[MilestoneFlow] = []
    for i, m in enumerate(milestones):
        timing = _milestone_timing_and_weight(
            m, params, i, market_entry, public_start,
            commercialization_weight, p_reimbursement, cumulative_sales, years,
        )
        if timing is None:
            continue
        year, weight = timing
        amount = float(params.get(f"milestone{i}_amount", m.amount_usd.base()))
        expected = amount * weight
        pv = present_value_at(rate, expected, year, convention)
        milestone_pv += pv
        flows.append(MilestoneFlow(label=m.label, year=year, expected_amount=expected, pv=pv))

    licensor_value = upfront_pv + milestone_pv + royalty_pv
    licensee_value = pie_rnpv - licensor_value

    return DealValuation(
        pie_rnpv=pie_rnpv,
        upfront_pv=upfront_pv,
        milestone_pv=milestone_pv,
        royalty_pv=royalty_pv,
        licensor_value=licensor_value,
        licensee_value=licensee_value,
        milestone_flows=flows,
    )
