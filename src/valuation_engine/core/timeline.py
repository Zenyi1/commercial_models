"""Discounting primitives.

A *timeline* is a sequence of annual periods indexed ``0, 1, 2, ...`` where
period ``p`` covers the calendar interval ``[p, p+1)`` measured in years from
the valuation date (``t = 0`` = "now"). A cash flow assigned to period ``p`` is
discounted by a factor that depends on the chosen ``convention``:

* ``"mid"``  – mid-year convention, exponent ``p + 0.5``. Standard for pharma
  DCF because revenue accrues roughly uniformly through the year rather than as
  a lump at year-end. This is the default.
* ``"end"``  – end-of-year convention, exponent ``p + 1``.

All functions here are pure and operate on plain floats / numpy arrays so they
can be checked against closed-form known answers in tests.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Convention = str  # "mid" | "end"

_VALID_CONVENTIONS = ("mid", "end")


def period_exponents(n_periods: int, convention: Convention = "mid") -> np.ndarray:
    """Return the discount exponent for each period ``0..n_periods-1``."""
    if n_periods < 0:
        raise ValueError("n_periods must be non-negative")
    p = np.arange(n_periods, dtype=float)
    if convention == "mid":
        return p + 0.5
    if convention == "end":
        return p + 1.0
    raise ValueError(f"convention must be one of {_VALID_CONVENTIONS}, got {convention!r}")


def discount_factors(
    rate: float, n_periods: int, convention: Convention = "mid"
) -> np.ndarray:
    """Vector of discount factors ``1 / (1+rate)**exponent`` per period.

    ``rate`` is a per-annum effective discount rate (e.g. ``0.12`` for 12%).
    """
    if rate <= -1.0:
        raise ValueError("discount rate must be greater than -1")
    exponents = period_exponents(n_periods, convention)
    return 1.0 / np.power(1.0 + rate, exponents)


def npv(
    rate: float, cashflows: Sequence[float] | np.ndarray, convention: Convention = "mid"
) -> float:
    """Net present value of a per-period cash-flow vector.

    ``cashflows[p]`` is the (already probability-adjusted, if applicable) net
    cash flow occurring in period ``p``.
    """
    cf = np.asarray(cashflows, dtype=float)
    if cf.ndim != 1:
        raise ValueError("cashflows must be one-dimensional")
    return float(np.dot(cf, discount_factors(rate, cf.shape[0], convention)))


def present_value_at(
    rate: float, amount: float, year: float, convention: Convention = "mid"
) -> float:
    """Discount a single ``amount`` occurring at fractional ``year`` from t=0.

    Used for one-off items whose timing is not aligned to an integer period,
    e.g. an upfront paid at signing (year 0) or a milestone paid mid-period.
    With ``convention="mid"`` an integer ``year`` is treated as the *start* of
    that period and shifted by +0.5; ``"end"`` shifts by +1. Pass the exact
    fractional year and ``convention="end"`` (exponent == year) for full
    control.
    """
    if rate <= -1.0:
        raise ValueError("discount rate must be greater than -1")
    if convention == "mid":
        exponent = year + 0.5
    elif convention == "end":
        exponent = year
    else:
        raise ValueError(f"convention must be one of {_VALID_CONVENTIONS}")
    return float(amount / (1.0 + rate) ** exponent)
