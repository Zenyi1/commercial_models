"""Pull sourced research into the model's asset x territory override structure.

``enrich_territory_asset`` queries a :class:`ResearchProvider` for each canonical
input and assembles a :class:`TerritoryAssetInputs`. Only fields the provider
actually answers are set; everything else falls back to territory-pack defaults
in the resolver. This is the join between the evidence layer and the model:
research fills the overrides, the resolver merges them, nothing is invented.

Each key is an independent network round-trip (Valyu search, then optional LLM
extraction), so the keys are fetched **concurrently** on a bounded thread pool —
wall-clock drops from sum-of-calls to slowest-call. The bound (``max_workers``)
keeps us polite to the Valyu / Anthropic rate limits; results are collected into
a dict keyed by field, so ordering is irrelevant.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

from valuation_engine.inputs.schemas import AssetSpec, TerritoryAssetInputs
from valuation_engine.research.provider import ResearchProvider, ResearchQuery

# Concurrency cap for per-key research calls. Modest by default so we don't trip
# provider rate limits; override with ENRICH_MAX_WORKERS.
_DEFAULT_MAX_WORKERS = int(os.getenv("ENRICH_MAX_WORKERS", "8"))

# Canonical keys that map 1:1 onto TerritoryAssetInputs SourcedValue fields.
_OVERRIDE_FIELDS = (
    "p_territory_approval",
    "p_reimbursement",
    "reimbursement_lag_years",
    "regulatory_review_years",
    "net_price_usd",
    "reference_price_factor",
    "peak_share",
    "launch_delay_years",
    "market_access_spend_usd",
    # Epidemiology funnel — now plumbed through TerritoryAssetInputs so a
    # researched local prevalence/diagnosis/treatment/eligibility figure reaches
    # the model instead of being dropped.
    "epi_rate_per_100k",
    "diagnosis_rate",
    "treatment_rate",
    "eligible_fraction",
)


def enrich_territory_asset(
    provider: ResearchProvider,
    asset: AssetSpec,
    territory_id: str,
    keys: Optional[Iterable[str]] = None,
    max_workers: Optional[int] = None,
) -> TerritoryAssetInputs:
    requested = _OVERRIDE_FIELDS if keys is None else tuple(keys)
    fields = [k for k in requested if k in _OVERRIDE_FIELDS]
    if not fields:
        return TerritoryAssetInputs()

    def fetch(key: str):
        q = ResearchQuery(
            key=key,
            territory_id=territory_id,
            asset_id=asset.id,
            indication=asset.indication,
            modality_id=asset.modality_id,
        )
        # provider.get is I/O-bound (network) and holds no shared mutable state
        # across calls — each ValyuProvider.get builds a fresh client — so it is
        # safe to run concurrently across threads.
        return key, provider.get(q)

    workers = max(1, min(max_workers or _DEFAULT_MAX_WORKERS, len(fields)))
    found: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, sv in pool.map(fetch, fields):
            if sv is not None:
                found[key] = sv
    return TerritoryAssetInputs(**found)
