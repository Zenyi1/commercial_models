"""Monte Carlo simulation and tornado sensitivity over a resolved input set.

``simulate`` pre-samples every parameter's distribution once (vectorised), then
runs the deterministic assembler once per draw. Binary gates (clinical launch,
territory approval, reimbursement) are realised 0/1 per draw, so the resulting
value distribution is genuinely bimodal where it should be — the honest picture
of territorial-rights risk. It reports P10/P50/P90 and the probability the
rights clear zero.

``tornado`` holds every parameter at its base and swings one at a time to its
P10/P90 (or 0/1 for a binary event), ranking the drivers by their impact on a
chosen target metric — the "what moves the number" chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from valuation_engine.inputs.resolve import ResolvedInputs
from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.rnpv import ModelResult, run_model

# Standard scalar metrics pulled from each model run.
Metric = Callable[[ModelResult], float]

STANDARD_METRICS: dict[str, Metric] = {
    "pie_rnpv": lambda r: r.pie_rnpv,
    "net_value": lambda r: r.net_value,
    "licensor_value": lambda r: r.deal.licensor_value,
    "licensee_value": lambda r: r.deal.licensee_value,
    "peak_net_sales": lambda r: r.peak_net_sales,
}


@dataclass
class Summary:
    p10: float
    p50: float
    p90: float
    mean: float
    std: float
    p_positive: float  # P(metric > 0)


@dataclass
class SimulationResult:
    n: int
    seed: int
    metrics: dict[str, np.ndarray]

    def summary(self, metric: str = "pie_rnpv") -> Summary:
        x = self.metrics[metric]
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        return Summary(
            p10=float(p10), p50=float(p50), p90=float(p90),
            mean=float(np.mean(x)), std=float(np.std(x)),
            p_positive=float(np.mean(x > 0.0)),
        )

    def all_summaries(self) -> dict[str, Summary]:
        return {name: self.summary(name) for name in self.metrics}


def _run_metrics(
    resolved: ResolvedInputs, params: Mapping[str, float], convention: str
) -> dict[str, float]:
    res = run_model(resolved, params, convention)
    return {name: fn(res) for name, fn in STANDARD_METRICS.items()}


def simulate(
    resolved: ResolvedInputs,
    n: int = 10_000,
    seed: int = 12345,
    convention: str = "mid",
) -> SimulationResult:
    """Run ``n`` Monte Carlo draws and collect the standard metrics."""
    rng = np.random.default_rng(seed)
    sampled = resolved.sample_params(rng, n)  # dict name -> array(n)
    keys = list(sampled.keys())

    out: dict[str, list[float]] = {name: [] for name in STANDARD_METRICS}
    for i in range(n):
        draw = {k: float(sampled[k][i]) for k in keys}
        vals = _run_metrics(resolved, draw, convention)
        for name, v in vals.items():
            out[name].append(v)

    return SimulationResult(
        n=n, seed=seed, metrics={name: np.asarray(v) for name, v in out.items()}
    )


# --------------------------------------------------------------------------- #
# Tornado sensitivity
# --------------------------------------------------------------------------- #
@dataclass
class TornadoBar:
    param: str
    low_input: float
    high_input: float
    target_low: float
    target_high: float

    @property
    def swing(self) -> float:
        return abs(self.target_high - self.target_low)


@dataclass
class TornadoResult:
    metric: str
    base_value: float
    bars: list[TornadoBar]  # sorted by swing, descending


def _swing_bounds(sv: SourcedValue, rng: np.random.Generator) -> tuple[float, float] | None:
    """Representative low/high input for a one-at-a-time swing, or None to skip."""
    if sv.kind == "point":
        return None
    if sv.kind == "bernoulli":
        return 0.0, 1.0  # show the full binary impact
    samples = sv.distribution().sample(rng, 20_000)
    lo, hi = np.percentile(samples, [10, 90])
    if np.isclose(lo, hi):
        return None
    return float(lo), float(hi)


def tornado(
    resolved: ResolvedInputs,
    metric: str = "pie_rnpv",
    seed: int = 7,
    convention: str = "mid",
    top: int | None = None,
) -> TornadoResult:
    """One-at-a-time sensitivity of ``metric`` to each uncertain parameter."""
    rng = np.random.default_rng(seed)
    base = resolved.base_params()
    fn = STANDARD_METRICS[metric]
    base_value = fn(run_model(resolved, base, convention))

    bars: list[TornadoBar] = []
    for name, sv in resolved.sourced.items():
        bounds = _swing_bounds(sv, rng)
        if bounds is None:
            continue
        lo_in, hi_in = bounds
        lo_target = fn(run_model(resolved, {**base, name: lo_in}, convention))
        hi_target = fn(run_model(resolved, {**base, name: hi_in}, convention))
        bars.append(
            TornadoBar(param=name, low_input=lo_in, high_input=hi_in,
                       target_low=lo_target, target_high=hi_target)
        )

    bars.sort(key=lambda b: b.swing, reverse=True)
    if top is not None:
        bars = bars[:top]
    return TornadoResult(metric=metric, base_value=base_value, bars=bars)
