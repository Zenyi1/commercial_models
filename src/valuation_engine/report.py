"""End-to-end territory valuation and reporting.

``value_territory`` ties the whole engine together for one asset x territory:

1. clinical LoA from the :class:`ClinicalModelProvider` (out-of-scope risk, injected);
2. optional research enrichment of asset x territory overrides (sourced);
3. resolve -> deterministic base case;
4. Monte Carlo (P10/P50/P90, P>0) + tornado;
5. comparables triangulation;
6. a go/no-go decision with expected value of basket inclusion and break-evens;
7. an assumptions ledger where every input traces to a source or explicit assumption.

``render_text`` formats it for the terminal — this is the number, its range, and
its justification, as it would go in front of a CEO / PE head. No UI beyond text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from valuation_engine.clinical.provider import ClinicalModelProvider, DefaultLoAProvider
from valuation_engine.comparables.analyzer import ComparablesAnalyzer
from valuation_engine.comparables.triangulation import Triangulation, triangulate
from valuation_engine.core.montecarlo import Summary, simulate, tornado
from valuation_engine.core.montecarlo import TornadoResult
from valuation_engine.inputs.resolve import ResolvedInputs, resolve
from valuation_engine.inputs.schemas import (
    AssetSpec,
    DealTerms,
    ModalityPack,
    SourcedValue,
    TerritoryAssetInputs,
    TerritoryPack,
)
from valuation_engine.research.enrich import enrich_territory_asset
from valuation_engine.research.provider import ResearchProvider
from valuation_engine.rnpv import ModelResult, run_model


@dataclass
class Decision:
    recommendation: str  # GO | CONDITIONAL | NO-GO
    base_net_value: float
    p50_net_value: float
    p_value_positive: float
    cost_to_enter_pv: float
    ev_basket_inclusion: float
    breakeven_p_reimbursement: str
    breakeven_reference_price_factor: str
    rationale: str


@dataclass
class TerritoryValuation:
    territory_id: str
    territory_name: str
    base: ModelResult
    summaries: dict[str, Summary]
    tornado: TornadoResult
    triangulation: Optional[Triangulation]
    decision: Decision
    ledger: list[tuple[str, SourcedValue]] = field(default_factory=list)
    n_mc: int = 0
    seed: int = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _merge_overrides(
    existing: TerritoryAssetInputs, researched: TerritoryAssetInputs
) -> TerritoryAssetInputs:
    """Explicit asset overrides win; research fills the gaps."""
    merged = researched.model_dump()
    for k, v in existing.model_dump().items():
        if v is not None and not (isinstance(v, list) and len(v) == 0):
            merged[k] = v
    return TerritoryAssetInputs.model_validate(merged)


def _net_value_at(resolved: ResolvedInputs, overrides: dict[str, float], convention: str) -> float:
    params = {**resolved.base_params(), **overrides}
    return run_model(resolved, params, convention).net_value


def _scan_breakeven(
    resolved: ResolvedInputs, param: str, grid: np.ndarray, convention: str
) -> Optional[float]:
    """First value of ``param`` on ``grid`` where net_value crosses zero (interpolated)."""
    prev_x, prev_y = None, None
    for x in grid:
        y = _net_value_at(resolved, {param: float(x)}, convention)
        if prev_y is not None and (prev_y < 0 <= y or prev_y <= 0 < y):
            # linear interpolation of the zero crossing
            if y != prev_y:
                return float(prev_x + (0 - prev_y) * (x - prev_x) / (y - prev_y))
            return float(x)
        prev_x, prev_y = x, y
    return None


def _breakeven_status(
    resolved: ResolvedInputs, param: str, grid: np.ndarray, convention: str, fmt
) -> str:
    """Human-readable break-even: the crossing value, or why there isn't one."""
    lo_y = _net_value_at(resolved, {param: float(grid[0])}, convention)
    hi_y = _net_value_at(resolved, {param: float(grid[-1])}, convention)
    if lo_y >= 0 and hi_y >= 0:
        return f"viable across full range (net value >=0 for all {param})"
    if lo_y < 0 and hi_y < 0:
        return f"not achievable within {param} in [{grid[0]:g}, {grid[-1]:g}]"
    crossing = _scan_breakeven(resolved, param, grid, convention)
    return fmt(crossing) if crossing is not None else "n/a"


def _ev_basket_inclusion(resolved: ResolvedInputs, convention: str) -> float:
    """Value attributable to making the reimbursement basket = pie(in) - pie(out)."""
    base = resolved.base_params()
    pie_in = run_model(resolved, {**base, "p_reimbursement": 1.0}, convention).pie_rnpv
    pie_out = run_model(resolved, {**base, "p_reimbursement": 0.0}, convention).pie_rnpv
    return pie_in - pie_out


def _decide(resolved: ResolvedInputs, base: ModelResult, summaries: dict[str, Summary],
            convention: str) -> Decision:
    nv = summaries["net_value"] if "net_value" in summaries else None
    p_pos = nv.p_positive if nv else 0.0
    p50 = nv.p50 if nv else base.net_value

    if p50 > 0 and p_pos >= 0.5:
        rec = "GO"
        rationale = "Median risk-adjusted rights value is positive with >50% chance of clearing entry cost."
    elif base.net_value > 0 or p_pos >= 0.3:
        rec = "CONDITIONAL"
        rationale = "Positive in expectation but outcome-dependent; hinges on launch and basket access."
    else:
        rec = "NO-GO"
        rationale = "Risk-adjusted value does not justify the cost to enter under current assumptions."

    return Decision(
        recommendation=rec,
        base_net_value=base.net_value,
        p50_net_value=p50,
        p_value_positive=p_pos,
        cost_to_enter_pv=base.cost_to_enter_pv,
        ev_basket_inclusion=_ev_basket_inclusion(resolved, convention),
        breakeven_p_reimbursement=_breakeven_status(
            resolved, "p_reimbursement", np.linspace(0.0, 1.0, 41), convention,
            lambda v: f"{v:.0%} reimbursement probability"),
        breakeven_reference_price_factor=_breakeven_status(
            resolved, "reference_price_factor", np.linspace(0.1, 1.5, 57), convention,
            lambda v: f"{v:.2f} price factor"),
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def value_territory(
    asset: AssetSpec,
    modality: ModalityPack,
    territory: TerritoryPack,
    clinical_provider: Optional[ClinicalModelProvider] = None,
    research_provider: Optional[ResearchProvider] = None,
    deal: Optional[DealTerms] = None,
    analyzer: Optional[ComparablesAnalyzer] = None,
    reference_value_usd: Optional[float] = None,
    n_mc: int = 10_000,
    seed: int = 12345,
    convention: str = "mid",
    tornado_metric: str = "pie_rnpv",
) -> TerritoryValuation:
    clinical_provider = clinical_provider or DefaultLoAProvider()
    loa = clinical_provider.probability_of_launch(asset)

    # Research enrichment -> merge into this territory's overrides.
    if research_provider is not None and research_provider.available():
        researched = enrich_territory_asset(research_provider, asset, territory.id)
        existing = asset.territory_overrides.get(territory.id, TerritoryAssetInputs())
        merged = _merge_overrides(existing, researched)
        overrides = dict(asset.territory_overrides)
        overrides[territory.id] = merged
        asset = asset.model_copy(update={"territory_overrides": overrides})

    resolved = resolve(asset, modality, territory, loa, deal)
    base = run_model(resolved, resolved.base_params(), convention)
    sim = simulate(resolved, n=n_mc, seed=seed, convention=convention)
    summaries = sim.all_summaries()
    torn = tornado(resolved, metric=tornado_metric, convention=convention, top=8)

    tri = None
    if analyzer is not None:
        tri = triangulate(
            base.pie_rnpv, base.peak_net_sales, analyzer, territory,
            reference_value_usd=reference_value_usd,
        )

    decision = _decide(resolved, base, summaries, convention)

    return TerritoryValuation(
        territory_id=territory.id,
        territory_name=territory.name,
        base=base,
        summaries=summaries,
        tornado=torn,
        triangulation=tri,
        decision=decision,
        ledger=resolved.ledger(),
        n_mc=n_mc,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Rendering (plain text / markdown for the terminal)
# --------------------------------------------------------------------------- #
def _m(x: float) -> str:
    return f"${x/1e6:,.1f}M"


def render_text(tv: TerritoryValuation, show_ledger: bool = True) -> str:
    s = tv.summaries
    pie, nv = s["pie_rnpv"], s["net_value"]
    lic, lce = s["licensor_value"], s["licensee_value"]
    d = tv.decision
    lines: list[str] = []
    add = lines.append

    add(f"{'='*72}")
    add(f"TERRITORY: {tv.territory_name}  ({tv.territory_id})")
    add(f"{'='*72}")
    add(f"Discount rate: {tv.base.rate:.1%}   Market entry: yr {tv.base.market_entry_year:.1f}   "
        f"Public/basket start: yr {tv.base.public_start_year:.1f}")
    add(f"Launch weight (clinical x territory approval): {tv.base.launch_weight:.2f}")
    add("")
    add("HEADLINE — territory rights value (risk-adjusted NPV of operating cash flow)")
    add(f"  Base case (mode):     {_m(tv.base.pie_rnpv)}")
    add(f"  Monte Carlo ({tv.n_mc:,} draws):  P10 {_m(pie.p10)}  |  P50 {_m(pie.p50)}  |  P90 {_m(pie.p90)}")
    add(f"  Mean {_m(pie.mean)}   P(value > 0): {pie.p_positive:.0%}")
    add(f"  Peak net sales (base): {_m(tv.base.peak_net_sales)}")
    add("")
    add("DEAL SPLIT (base case)")
    add(f"  Licensor value: {_m(tv.base.deal.licensor_value)}   "
        f"(upfront {_m(tv.base.deal.upfront_pv)} + milestones {_m(tv.base.deal.milestone_pv)} "
        f"+ royalties {_m(tv.base.deal.royalty_pv)})")
    add(f"  Licensee value: {_m(tv.base.deal.licensee_value)}   "
        f"[MC licensor P50 {_m(lic.p50)}, licensee P50 {_m(lce.p50)}]")
    add("")
    add("DECISION — is it worth entering?")
    add(f"  Recommendation: {d.recommendation}")
    add(f"  Net value (rights - cost to enter): base {_m(d.base_net_value)}, "
        f"MC P50 {_m(d.p50_net_value)}, P(>0) {d.p_value_positive:.0%}")
    add(f"  Cost to enter (PV): {_m(d.cost_to_enter_pv)}")
    add(f"  Expected value of basket inclusion: {_m(d.ev_basket_inclusion)}")
    add(f"  Break-even reimbursement: {d.breakeven_p_reimbursement}")
    add(f"  Break-even price: {d.breakeven_reference_price_factor}")
    add(f"  Rationale: {d.rationale}")
    add("")
    add(f"TOP VALUE DRIVERS (tornado on {tv.tornado.metric}, base {_m(tv.tornado.base_value)})")
    for b in tv.tornado.bars[:6]:
        add(f"  {b.param:26s} swing {_m(b.swing):>10s}   "
            f"[{_m(b.target_low)} .. {_m(b.target_high)}]")
    add("")
    if tv.triangulation is not None:
        add("TRIANGULATION (independent cross-checks)")
        for leg in tv.triangulation.legs:
            v = _m(leg.value) if leg.value is not None else "n/a"
            add(f"  {leg.name:12s} {v:>10s}   ({leg.method})")
        flag = "  FLAGGED — legs diverge" if tv.triangulation.flagged else "  consistent"
        add(f"  Reconciled: {_m(tv.triangulation.reconciled)}   "
            f"spread {tv.triangulation.spread_ratio:.1f}x  ->{flag}")
        for n in tv.triangulation.notes:
            add(f"    note: {n}")
        add("")
    if show_ledger:
        add("ASSUMPTIONS LEDGER (every input traces to a source or explicit assumption)")
        for name, sv in tv.ledger:
            rng = ""
            if sv.low is not None or sv.high is not None:
                rng = f" [{sv.low}–{sv.high}]"
            conf = sv.provenance.confidence
            src = sv.provenance.source or sv.provenance.method or "—"
            add(f"  {name:28s} {sv.value:>12,.4g}{rng:16s} ({conf:10s}) {src}")
    return "\n".join(lines)


def render_portfolio(tvs: list[TerritoryValuation]) -> str:
    lines = ["", "#" * 72, "PORTFOLIO SUMMARY — territory rights value", "#" * 72,
             f"{'Territory':16s} {'Base rNPV':>12s} {'P10':>10s} {'P50':>10s} {'P90':>10s} "
             f"{'P(>0)':>7s} {'Decision':>12s}"]
    total_p50 = 0.0
    for tv in tvs:
        pie = tv.summaries["pie_rnpv"]
        total_p50 += pie.p50
        lines.append(
            f"{tv.territory_name:16s} {_m(tv.base.pie_rnpv):>12s} {_m(pie.p10):>10s} "
            f"{_m(pie.p50):>10s} {_m(pie.p90):>10s} {pie.p_positive:>6.0%} "
            f"{tv.decision.recommendation:>12s}"
        )
    lines.append("-" * 72)
    lines.append(f"{'Sum of P50':16s} {'':12s} {'':10s} {_m(total_p50):>10s}")
    return "\n".join(lines)
