"""Benchmark likelihood-of-approval (LoA) table — a placeholder for the user's
clinical model.

``BASE_LOA[phase]`` is the cumulative probability of reaching regulatory
approval measured from the *start* of that development phase, using published
industry success-rate benchmarks (Wong, Siah & Lo 2019; BIO/Biomedtracker
Clinical Development Success Rates). ``MODALITY_FACTOR`` applies a coarse
relative adjustment by modality. Results are clamped to [0, 1].

These values are deliberately conservative and clearly low-confidence: they
exist so the engine runs end-to-end before the user's own clinical PoS models
are wired in via :class:`~valuation_engine.clinical.provider.ClinicalModelProvider`.
"""

from __future__ import annotations

# Cumulative LoA from the start of each phase to approval.
BASE_LOA: dict[str, float] = {
    "preclinical": 0.06,
    "phase1": 0.10,
    "phase2": 0.17,
    "phase3": 0.55,
    "filed": 0.88,
    "approved": 1.0,
}

# Coarse relative modality multipliers (clamped result in [0,1]).
MODALITY_FACTOR: dict[str, float] = {
    "small_molecule": 1.00,
    "biologic": 1.05,
    "gene_therapy": 0.90,  # newer modality, higher development uncertainty
}

_SOURCE = "https://doi.org/10.1093/biostatistics/kxx069"
_PUBLISHER = "Wong, Siah & Lo (2019), Biostatistics; BIO Clinical Development Success Rates"


def default_loa(phase: str, modality_id: str) -> tuple[float, dict]:
    """Return ``(loa_value, provenance_dict)`` for a phase and modality."""
    if phase not in BASE_LOA:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(BASE_LOA)}")
    base = BASE_LOA[phase]
    factor = MODALITY_FACTOR.get(modality_id, 1.0)
    value = max(0.0, min(1.0, base * factor))
    provenance = {
        "source": _SOURCE,
        "publisher": _PUBLISHER,
        "method": f"Benchmark cumulative LoA from {phase} x modality factor {factor}",
        "confidence": "low",
        "notes": "Placeholder industry benchmark; replace with the user's clinical PoS model.",
    }
    return value, provenance
