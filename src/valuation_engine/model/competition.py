"""Competition: differentiation sets the share ceiling; the in-territory
competitor pipeline erodes share dynamically as rivals launch.

* ``adjusted_peak_share`` — scales the asset's intrinsic peak share by its
  differentiation vs the standard of care *available in that market*
  (``differentiation_score``: 1 = parity, >1 better, <1 worse), capped.
* ``competition_retention`` — per-year multiplier ``1 - Σ active competitor
  captures`` (floored at 0). A competitor only bites once it has launched
  in-territory, so a rival arriving in year 4 leaves the early years intact —
  directly answering "is another drug arriving at the same time?".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MAX_SHARE = 0.9


def adjusted_peak_share(
    peak_share: float, differentiation_score: float, cap: float = MAX_SHARE
) -> float:
    return float(min(max(peak_share * differentiation_score, 0.0), cap))


def competition_retention(
    years: np.ndarray,
    competitor_launch_years: Sequence[float],
    competitor_share_captures: Sequence[float],
) -> np.ndarray:
    """Per-period retained share fraction after competitor entry."""
    retention = np.ones(len(years), dtype=float)
    midyear = years + 0.5
    for launch, capture in zip(competitor_launch_years, competitor_share_captures):
        active = midyear >= launch
        retention = retention - active * capture
    return np.clip(retention, 0.0, 1.0)
