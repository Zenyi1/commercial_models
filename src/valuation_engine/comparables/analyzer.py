"""Regress the deal corpus into benchmarks and geography-adjustment factors.

Because territory-specific (LatAm/MENA) deal data is sparse, the analyzer mines
the whole global corpus and derives:

* **benchmark ranges** (P25/P50/P75) for royalty, upfront and total value,
  filterable by modality x stage x region;
* a **sales-to-value multiple** (deal value / disclosed peak sales) for the
  top-down cross-check;
* **geography-adjustment factors** — a market-size proxy that maps a global/US
  value onto a specific territory using population x health spend x price level;
* **territorial-split priors** — the empirical fraction of global deal value that
  regional (LatAm/MENA/…) deals have historically carried.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from valuation_engine.comparables.schema import DealRecord
from valuation_engine.inputs.schemas import TerritoryPack

# US market-size reference proxy (population x health expenditure/capita x price
# level = 1.0). Used only as the denominator for the geography-adjustment factor.
US_REFERENCE = {
    "population": 335_000_000.0,
    "health_exp_per_capita_usd": 12_500.0,
    "reference_price_factor": 1.0,
}

# Fallback territorial split priors (fraction of global deal value) when the
# corpus lacks enough regional comps for a stable estimate.
_FALLBACK_REGION_SPLIT = {
    "latam": 0.06, "mena": 0.04, "apac": 0.15, "china": 0.12, "eu": 0.30, "ex_us": 0.45,
}


@dataclass
class Benchmark:
    field: str
    n: int
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    mean: Optional[float]


def _stats(field: str, values: Sequence[float]) -> Benchmark:
    vals = [v for v in values if v is not None]
    if not vals:
        return Benchmark(field=field, n=0, p25=None, p50=None, p75=None, mean=None)
    arr = np.asarray(vals, dtype=float)
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    return Benchmark(field=field, n=len(vals), p25=float(p25), p50=float(p50),
                     p75=float(p75), mean=float(arr.mean()))


class ComparablesAnalyzer:
    def __init__(self, records: list[DealRecord]):
        self.records = records

    def filter(
        self,
        modality: Optional[str] = None,
        stage: Optional[str] = None,
        region: Optional[str] = None,
        deal_type: Optional[str] = None,
    ) -> list[DealRecord]:
        out = self.records
        if modality:
            out = [r for r in out if r.modality == modality]
        if stage:
            out = [r for r in out if r.stage == stage]
        if region:
            out = [r for r in out if region in r.regions]
        if deal_type:
            out = [r for r in out if r.deal_type == deal_type]
        return out

    def benchmark(self, field: str, **filters) -> Benchmark:
        recs = self.filter(**filters)
        return _stats(field, [getattr(r, field) for r in recs])

    def royalty_benchmark(self, **filters) -> Benchmark:
        return self.benchmark("peak_royalty_rate", **filters)

    def upfront_benchmark(self, **filters) -> Benchmark:
        return self.benchmark("upfront_usd", **filters)

    def sales_to_value_multiple(self) -> Optional[float]:
        """Median deal value / disclosed peak sales, across records disclosing both."""
        ratios = [
            r.total_value_usd / r.disclosed_peak_sales_usd
            for r in self.records
            if r.total_value_usd and r.disclosed_peak_sales_usd
        ]
        return float(np.median(ratios)) if ratios else None

    def geography_adjustment_factor(self, territory: TerritoryPack) -> float:
        """Territory value as a fraction of the US reference, via a market-size proxy.

        proxy = population x health-expenditure/capita x reference-price level.
        Captures how much smaller (or, rarely, larger) the territory opportunity
        is than the US — the transfer function for projecting global comps down.
        """
        t_proxy = (
            territory.population.base()
            * territory.macro.health_exp_per_capita_usd.base()
            * territory.reference_price_factor.base()
        )
        us_proxy = (
            US_REFERENCE["population"]
            * US_REFERENCE["health_exp_per_capita_usd"]
            * US_REFERENCE["reference_price_factor"]
        )
        return t_proxy / us_proxy if us_proxy > 0 else 0.0

    def territorial_split_prior(self, region: str, min_n: int = 2) -> float:
        """Empirical fraction of global deal value carried by a region's deals.

        Uses median regional total value / median global total value when enough
        regional comps exist; otherwise a documented fallback prior.
        """
        regional = [r.total_value_usd for r in self.filter(region=region)
                    if r.total_value_usd and r.deal_type != "global_license"]
        global_vals = [r.total_value_usd for r in self.filter(deal_type="global_license")
                       if r.total_value_usd]
        if len(regional) >= min_n and global_vals:
            return float(np.median(regional) / np.median(global_vals))
        return _FALLBACK_REGION_SPLIT.get(region, 0.05)

    def implied_territory_value(
        self, territory: TerritoryPack, reference_value_usd: float
    ) -> float:
        """Map a global/US reference value onto the territory via the geography factor."""
        return reference_value_usd * self.geography_adjustment_factor(territory)
