"""Impute missing royalty rates from disclosed comparables — honestly.

Most filings redact the royalty rate. Rather than leave those deals unusable (or,
worse, invent a number), this imputes the rate from the subset of deals where it
*is* disclosed, and returns a **distribution** (P25/P50/P75 as a PERT) that is
always tagged ``imputed`` with lower confidence and a note. An imputed rate is
never presented as a disclosed one.

Method — hierarchical stratified benchmark. Try the tightest stratum first and
fall back until a stratum has at least ``min_n`` disclosed comps:

    (stage, modality)  ->  (stage)  ->  (modality)  ->  all disclosed

If even the full disclosed set is too thin, fall back to a documented
industry stage-based prior (a rule-of-thumb band, flagged ``assumption``). This
keeps every imputed value traceable to either real comps or a named heuristic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from valuation_engine.comparables.schema import DealRecord
from valuation_engine.inputs.schemas import Provenance, SourcedValue

# Industry rule-of-thumb top-tier royalty bands by stage (low, base, high).
# Consistent with a ~8% median industry royalty and stage escalation; used only
# when disclosed comparables are too sparse to stratify.
INDUSTRY_STAGE_PRIOR: dict[str, tuple[float, float, float]] = {
    "preclinical": (0.03, 0.05, 0.08),
    "phase1": (0.05, 0.07, 0.10),
    "phase2": (0.07, 0.10, 0.13),
    "phase3": (0.10, 0.13, 0.17),
    "filed": (0.12, 0.15, 0.19),
    "approved": (0.12, 0.17, 0.22),
    "unknown": (0.05, 0.10, 0.15),
}


class RoyaltyImputer:
    """Impute royalty rates from a corpus's disclosed subset."""

    def __init__(self, records: list[DealRecord], min_n: int = 3,
                 prior: Optional[dict] = None):
        self.disclosed = [r for r in records if r.peak_royalty_rate is not None]
        self.min_n = min_n
        self.prior = prior or INDUSTRY_STAGE_PRIOR

    def _rates(self, stage: Optional[str], modality: Optional[str]) -> list[float]:
        return [
            r.peak_royalty_rate for r in self.disclosed
            if (stage is None or r.stage == stage)
            and (modality is None or r.modality == modality)
        ]

    def impute(self, stage: str = "unknown", modality: str = "other") -> SourcedValue:
        """Return an imputed royalty as a PERT SourcedValue (P25/P50/P75)."""
        strata = [
            (stage, modality, f"stage={stage}, modality={modality}"),
            (stage, None, f"stage={stage}"),
            (None, modality, f"modality={modality}"),
            (None, None, "all disclosed deals"),
        ]
        for s, m, desc in strata:
            rates = self._rates(s, m)
            if len(rates) >= self.min_n:
                p25, p50, p75 = np.percentile(rates, [25, 50, 75])
                return SourcedValue(
                    value=float(p50), low=float(p25), high=float(p75), kind="pert",
                    provenance=Provenance(
                        method=f"Imputed from {len(rates)} disclosed comparables ({desc})",
                        confidence="low",
                        notes="IMPUTED royalty — not a disclosed rate; from comparable-deal distribution.",
                    ),
                )
        lo, base, hi = self.prior.get(stage, self.prior["unknown"])
        return SourcedValue(
            value=base, low=lo, high=hi, kind="pert",
            provenance=Provenance(
                method=f"Industry stage-based royalty prior (stage={stage})",
                confidence="assumption",
                notes="IMPUTED from industry rule-of-thumb; no comparable disclosed rates available.",
            ),
        )

    def augment(self, records: list[DealRecord]) -> list[DealRecord]:
        """Return a copy of ``records`` with missing royalty rates imputed.

        Disclosed rates are left untouched. Imputed rows have their
        ``peak_royalty_rate`` filled with the imputed median, ``confidence``
        downgraded, and a note appended so imputed and disclosed never blur.
        """
        out: list[DealRecord] = []
        for r in records:
            if r.peak_royalty_rate is None:
                sv = self.impute(r.stage, r.modality)
                note = " | royalty imputed (" + (sv.provenance.method or "") + ")"
                out.append(r.model_copy(update={
                    "peak_royalty_rate": sv.value,
                    "confidence": "low",
                    "notes": (r.notes or "") + note,
                }))
            else:
                out.append(r)
        return out

    def coverage(self) -> dict:
        """Diagnostics: how much real royalty data backs the imputer."""
        return {"disclosed_rates": len(self.disclosed)}
