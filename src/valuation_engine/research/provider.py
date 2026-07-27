"""Research provider interface and multi-source aggregator."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

from valuation_engine.inputs.schemas import Confidence, SourcedValue

# Canonical input names a provider can answer. These mirror the resolver's
# parameter names / TerritoryAssetInputs fields so results can be applied
# directly. Not exhaustive — providers may answer any subset.
CANONICAL_KEYS = (
    "p_reimbursement",
    "p_territory_approval",
    "reimbursement_lag_years",
    "regulatory_review_years",
    "reference_price_factor",
    "net_price_usd",
    "peak_share",
    "launch_delay_years",
    "epi_rate_per_100k",
    "diagnosis_rate",
    "treatment_rate",
    "eligible_fraction",
)

_CONFIDENCE_RANK: dict[Confidence, int] = {
    "high": 3, "medium": 2, "low": 1, "assumption": 0,
}


@dataclass(frozen=True)
class ResearchQuery:
    """A request for one canonical input for a specific asset x territory."""

    key: str
    territory_id: str
    asset_id: Optional[str] = None
    indication: Optional[str] = None
    modality_id: Optional[str] = None
    context: dict = field(default_factory=dict)


class ResearchProvider(abc.ABC):
    """Answers a :class:`ResearchQuery` with a ``SourcedValue`` (or ``None``)."""

    name: str = "provider"

    def available(self) -> bool:
        """Whether this provider can currently be queried (keys/config present)."""
        return True

    @abc.abstractmethod
    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        """Return a sourced value for the query, or None if it can't answer."""


def _rank(sv: SourcedValue) -> int:
    return _CONFIDENCE_RANK.get(sv.provenance.confidence, 0)


class MultiSourceProvider(ResearchProvider):
    """Aggregate several providers; cross-validate rather than average.

    Strategy: query every *available* provider, keep the non-None candidates,
    and return the highest-confidence one. If the top two disagree materially,
    the returned value's provenance notes the disagreement and its confidence is
    downgraded — surfacing conflict instead of hiding it in a mean.
    """

    def __init__(
        self, providers: list[ResearchProvider], disagreement_tol: float = 0.25
    ):
        self.providers = providers
        self.disagreement_tol = disagreement_tol
        self.name = "multi(" + ",".join(p.name for p in providers) + ")"

    def available(self) -> bool:
        return any(p.available() for p in self.providers)

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        candidates: list[tuple[str, SourcedValue]] = []
        for p in self.providers:
            if not p.available():
                continue
            try:
                sv = p.get(query)
            except Exception:  # a flaky backend must not sink the whole query
                sv = None
            if sv is not None:
                candidates.append((p.name, sv))

        if not candidates:
            return None
        candidates.sort(key=lambda c: _rank(c[1]), reverse=True)
        best_name, best = candidates[0]
        if len(candidates) == 1:
            return best

        # Cross-validate against the next best.
        second_name, second = candidates[1]
        base = abs(best.value)
        rel = abs(best.value - second.value) / base if base > 0 else abs(best.value - second.value)
        if rel > self.disagreement_tol:
            note = (
                f"cross-source disagreement: {best_name}={best.value} vs "
                f"{second_name}={second.value} (rel {rel:.0%}); using higher-confidence source, "
                "confidence downgraded."
            )
            prov = best.provenance.model_copy(
                update={
                    "confidence": "low",
                    "notes": " | ".join(filter(None, [best.provenance.notes, note])),
                }
            )
            return best.model_copy(update={"provenance": prov})
        return best
