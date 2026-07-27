"""Probability distributions for uncertain inputs.

Every uncertain model input is expressed as a :class:`Distribution`. Two things
are asked of a distribution:

* ``base()``   – the single point value used for the deterministic base case.
* ``sample(rng, size)`` – draws for Monte Carlo.

Design choices for defensibility:

* **Three-point elicitation.** Most inputs are elicited as ``(low, base,
  high)`` and modelled with a **PERT** (a.k.a. beta-PERT) distribution. PERT is
  the standard choice for expert-elicited quantities: it is bounded by the
  low/high, peaks at the mode, and is smoother / less sensitive to the extremes
  than a triangular. ``low``/``high`` are treated as ~P5/P95-style plausibility
  bounds, not hard 0/100 percentiles — the classic project-estimation reading.
* **Binary events** (regulatory approval in-territory, formulary/basket
  inclusion) are :class:`Bernoulli`. In the base case they contribute their
  probability as an expected-value weight; in Monte Carlo they are realised as
  0/1, which is what produces a realistic, often bimodal, value distribution
  (rights are worth a lot if approved & reimbursed, ~nothing otherwise).
* The deterministic ``base()`` of a PERT is its **mode**, not its mean, so the
  headline base case reflects the analyst's single most-likely input. The mean
  differs from the mode for skewed three-point inputs; both are exposed.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

Number = float


class Distribution(abc.ABC):
    """Abstract uncertain scalar."""

    @abc.abstractmethod
    def base(self) -> float:
        """Point value for the deterministic base case."""

    @abc.abstractmethod
    def mean(self) -> float:
        """Analytic (or near-analytic) mean of the distribution."""

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` independent samples."""


@dataclass(frozen=True)
class Deterministic(Distribution):
    """A known constant — no uncertainty."""

    value: float

    def base(self) -> float:
        return float(self.value)

    def mean(self) -> float:
        return float(self.value)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, float(self.value))


@dataclass(frozen=True)
class Triangular(Distribution):
    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError(f"require low <= mode <= high, got {self.low}, {self.mode}, {self.high}")

    def base(self) -> float:
        return float(self.mode)

    def mean(self) -> float:
        return float((self.low + self.mode + self.high) / 3.0)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.low == self.high:
            return np.full(size, float(self.mode))
        return rng.triangular(self.low, self.mode, self.high, size=size)


@dataclass(frozen=True)
class PERT(Distribution):
    """Beta-PERT distribution from a three-point ``(low, mode, high)`` estimate.

    ``lamb`` (default 4) controls how much weight the mode carries; 4 is the
    canonical PERT value. Reduces to a constant when ``low == high``.
    """

    low: float
    mode: float
    high: float
    lamb: float = 4.0

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError(f"require low <= mode <= high, got {self.low}, {self.mode}, {self.high}")
        if self.lamb < 0:
            raise ValueError("lamb must be non-negative")

    def base(self) -> float:
        return float(self.mode)

    def mean(self) -> float:
        return float((self.low + self.lamb * self.mode + self.high) / (self.lamb + 2.0))

    def _beta_params(self) -> tuple[float, float]:
        span = self.high - self.low
        mean = self.mean()
        # Standard beta-PERT parameterisation.
        alpha = 1.0 + self.lamb * (self.mode - self.low) / span
        beta = 1.0 + self.lamb * (self.high - self.mode) / span
        # Guard against degenerate params when mode sits on a bound.
        alpha = max(alpha, 1e-6)
        beta = max(beta, 1e-6)
        _ = mean  # kept for readability / debugging parity
        return alpha, beta

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.high == self.low:
            return np.full(size, float(self.mode))
        alpha, beta = self._beta_params()
        return self.low + rng.beta(alpha, beta, size=size) * (self.high - self.low)


@dataclass(frozen=True)
class Beta(Distribution):
    """Beta distribution on ``[0, 1]`` — for probabilities with a mode/mean and
    a concentration. Built from ``mean`` and ``concentration`` (``alpha+beta``);
    higher concentration = tighter around the mean.
    """

    mean_: float
    concentration: float = 20.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.mean_ <= 1.0):
            raise ValueError("mean must be in [0, 1]")
        if self.concentration <= 0:
            raise ValueError("concentration must be positive")

    @property
    def alpha(self) -> float:
        return self.mean_ * self.concentration

    @property
    def beta_(self) -> float:
        return (1.0 - self.mean_) * self.concentration

    def base(self) -> float:
        return float(self.mean_)

    def mean(self) -> float:
        return float(self.mean_)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        a = max(self.alpha, 1e-6)
        b = max(self.beta_, 1e-6)
        return rng.beta(a, b, size=size)


@dataclass(frozen=True)
class Bernoulli(Distribution):
    """A binary event with success probability ``p``.

    ``base()`` returns ``p`` (used as an expected-value weight in the
    deterministic case). ``sample`` returns 0.0/1.0 realisations.
    """

    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be in [0, 1]")

    def base(self) -> float:
        return float(self.p)

    def mean(self) -> float:
        return float(self.p)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return (rng.random(size) < self.p).astype(float)


@dataclass(frozen=True)
class Lognormal(Distribution):
    """Right-skewed positive quantity specified by ``median`` and geometric
    ``sigma`` (multiplicative volatility). Useful for prices and market sizes,
    which are bounded below by zero and skewed right.
    """

    median: float
    sigma: float

    def __post_init__(self) -> None:
        if self.median <= 0:
            raise ValueError("median must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def base(self) -> float:
        return float(self.median)

    def mean(self) -> float:
        return float(self.median * np.exp(self.sigma**2 / 2.0))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        mu = np.log(self.median)
        return rng.lognormal(mean=mu, sigma=self.sigma, size=size)
