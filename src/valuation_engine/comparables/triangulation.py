"""Triangulate the territory value three independent ways and flag divergence.

A single method can be wrong in a way that isn't obvious. Triangulation reconciles:

1. **bottom-up** — the rNPV of operating cash flow from the model;
2. **top-down** — risked peak net sales x an rNPV-to-peak-sales multiple
   (a documented industry heuristic, independent of the discounted build-up);
3. **comparables-implied** — a global/US reference deal value projected onto the
   territory via the analyzer's geography-adjustment factor (only when a
   reference value is supplied).

If the legs disagree by more than ``threshold`` x (max/min), the result is
flagged so the number is scrutinised rather than trusted blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from valuation_engine.comparables.analyzer import ComparablesAnalyzer
from valuation_engine.inputs.schemas import TerritoryPack

# Documented heuristic: risk-adjusted NPV as a multiple of (risked) peak annual
# net sales. ~2-3x is a common rule of thumb for a de-risked in-market asset.
DEFAULT_VALUE_TO_PEAK_MULTIPLE = 2.5


@dataclass
class TriangulationLeg:
    name: str
    value: Optional[float]
    method: str


@dataclass
class Triangulation:
    legs: list[TriangulationLeg]
    reconciled: float
    spread_ratio: float
    flagged: bool
    threshold: float
    notes: list[str] = field(default_factory=list)


def triangulate(
    bottom_up_value: float,
    risked_peak_net_sales: float,
    analyzer: Optional[ComparablesAnalyzer] = None,
    territory: Optional[TerritoryPack] = None,
    reference_value_usd: Optional[float] = None,
    value_to_peak_multiple: float = DEFAULT_VALUE_TO_PEAK_MULTIPLE,
    threshold: float = 2.5,
) -> Triangulation:
    legs: list[TriangulationLeg] = [
        TriangulationLeg("bottom_up", bottom_up_value, "rNPV of operating cash flow (model)")
    ]

    top_down = risked_peak_net_sales * value_to_peak_multiple
    legs.append(TriangulationLeg(
        "top_down", top_down,
        f"risked peak net sales x {value_to_peak_multiple:.1f} rNPV-to-peak multiple",
    ))

    notes: list[str] = []
    if analyzer is not None and territory is not None and reference_value_usd is not None:
        comp = analyzer.implied_territory_value(territory, reference_value_usd)
        factor = analyzer.geography_adjustment_factor(territory)
        legs.append(TriangulationLeg(
            "comparables", comp,
            f"reference ${reference_value_usd/1e6:.0f}M x geography factor {factor:.3f}",
        ))
    else:
        notes.append("comparables leg omitted (no reference value supplied)")

    present = [l.value for l in legs if l.value is not None]
    reconciled = float(np.median(present)) if present else 0.0

    positive = [v for v in present if v > 0]
    if len(positive) >= 2 and min(positive) > 0:
        spread_ratio = max(positive) / min(positive)
    else:
        spread_ratio = 1.0
    flagged = spread_ratio > threshold
    if flagged:
        notes.append(
            f"divergence: legs span {spread_ratio:.1f}x (> {threshold:.1f}x) — scrutinise assumptions"
        )

    return Triangulation(
        legs=legs, reconciled=reconciled, spread_ratio=spread_ratio,
        flagged=flagged, threshold=threshold, notes=notes,
    )
