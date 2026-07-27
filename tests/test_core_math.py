"""Known-answer tests for the domain-agnostic core math.

These pin down the discounting and distribution primitives against
hand-computed closed-form values so downstream model changes can't silently
corrupt the financial arithmetic.
"""

import math

import numpy as np
import pytest

from valuation_engine.core import distributions as D
from valuation_engine.core import timeline as T


# --------------------------------------------------------------------------- #
# Discounting
# --------------------------------------------------------------------------- #
def test_discount_factor_end_convention():
    # $100 one year out at 10% end-of-year -> 90.909...
    df = T.discount_factors(0.10, 1, convention="end")
    assert df[0] == pytest.approx(1 / 1.10)


def test_discount_factor_mid_convention():
    df = T.discount_factors(0.10, 1, convention="mid")
    assert df[0] == pytest.approx(1 / 1.10**0.5)


def test_npv_zero_rate_is_sum():
    assert T.npv(0.0, [10.0, 20.0, 30.0]) == pytest.approx(60.0)


def test_npv_known_answer_end_convention():
    # 100 at end of year 1, 100 at end of year 2, at 10%.
    val = T.npv(0.10, [100.0, 100.0], convention="end")
    assert val == pytest.approx(100 / 1.10 + 100 / 1.10**2)


def test_npv_mid_greater_than_end():
    # Mid-year cash arrives earlier -> discounted less -> larger PV.
    cf = [100.0] * 5
    assert T.npv(0.12, cf, convention="mid") > T.npv(0.12, cf, convention="end")


def test_present_value_at_signing_is_undiscounted_end():
    # Upfront paid "now" (year 0) with end convention has exponent 0.
    assert T.present_value_at(0.12, 5.0, 0.0, convention="end") == pytest.approx(5.0)


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        T.discount_factors(-1.0, 3)


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def test_pert_mean_formula():
    d = D.PERT(low=0.0, mode=50.0, high=100.0)
    # (0 + 4*50 + 100) / 6 = 50
    assert d.mean() == pytest.approx(50.0)
    assert d.base() == 50.0


def test_pert_skewed_mean():
    d = D.PERT(low=0.0, mode=10.0, high=100.0)
    assert d.mean() == pytest.approx((0 + 4 * 10 + 100) / 6.0)
    assert d.base() == 10.0  # base is the mode, not the mean


def test_pert_samples_within_bounds_and_track_mean():
    rng = np.random.default_rng(42)
    d = D.PERT(low=2.0, mode=5.0, high=20.0)
    s = d.sample(rng, 200_000)
    assert s.min() >= 2.0 and s.max() <= 20.0
    assert s.mean() == pytest.approx(d.mean(), rel=0.02)


def test_pert_degenerate_when_low_equals_high():
    rng = np.random.default_rng(0)
    d = D.PERT(low=7.0, mode=7.0, high=7.0)
    assert np.all(d.sample(rng, 100) == 7.0)


def test_triangular_mean():
    d = D.Triangular(0.0, 3.0, 6.0)
    assert d.mean() == pytest.approx(3.0)


def test_bernoulli_sample_mean_tracks_p():
    rng = np.random.default_rng(7)
    d = D.Bernoulli(0.3)
    s = d.sample(rng, 500_000)
    assert set(np.unique(s)).issubset({0.0, 1.0})
    assert s.mean() == pytest.approx(0.3, abs=0.005)
    assert d.base() == 0.3


def test_beta_probability_mean():
    rng = np.random.default_rng(1)
    d = D.Beta(mean_=0.65, concentration=40)
    s = d.sample(rng, 300_000)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s.mean() == pytest.approx(0.65, abs=0.01)


def test_lognormal_median_and_mean():
    d = D.Lognormal(median=100.0, sigma=0.5)
    assert d.base() == 100.0
    assert d.mean() == pytest.approx(100.0 * math.exp(0.25 / 2))


def test_invalid_pert_ordering_raises():
    with pytest.raises(ValueError):
        D.PERT(low=10.0, mode=5.0, high=20.0)
