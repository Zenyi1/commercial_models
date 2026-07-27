"""Uptake curve — fraction of peak reached over time since market entry.

A smootherstep S-curve (``6t^5 - 15t^4 + 10t^3``) rising from 0 at entry to 1 at
``years_to_peak``. Smoother than a linear ramp and than triangular, with zero
slope at both ends, which matches how launch uptake actually accelerates then
saturates. For recurring therapies this scales the market share; for one_time
therapies it scales the annual capture rate of the treatable pool.
"""

from __future__ import annotations

import numpy as np


def uptake_fraction(
    years: np.ndarray, market_entry: float, years_to_peak: float
) -> np.ndarray:
    midyear = years + 0.5
    elapsed = np.maximum(0.0, midyear - market_entry)
    t = np.clip(elapsed / max(years_to_peak, 1e-9), 0.0, 1.0)
    # Smootherstep (Ken Perlin): 6t^5 - 15t^4 + 10t^3.
    s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    # Only active from entry onward.
    return np.where(midyear >= market_entry, s, 0.0)
