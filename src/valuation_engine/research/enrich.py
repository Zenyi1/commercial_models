"""Pull sourced research into the model's asset x territory override structure.

``enrich_territory_asset`` queries a :class:`ResearchProvider` for each canonical
input and assembles a :class:`TerritoryAssetInputs`. Only fields the provider
actually answers are set; everything else falls back to territory-pack defaults
in the resolver. This is the join between the evidence layer and the model:
research fills the overrides, the resolver merges them, nothing is invented.
"""

from __future__ import annotations

from typing import Iterable, Optional

from valuation_engine.inputs.schemas import AssetSpec, TerritoryAssetInputs
from valuation_engine.research.provider import ResearchProvider, ResearchQuery

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
) -> TerritoryAssetInputs:
    keys = tuple(keys) if keys is not None else _OVERRIDE_FIELDS
    found: dict = {}
    for key in keys:
        if key not in _OVERRIDE_FIELDS:
            continue
        q = ResearchQuery(
            key=key,
            territory_id=territory_id,
            asset_id=asset.id,
            indication=asset.indication,
            modality_id=asset.modality_id,
        )
        sv = provider.get(q)
        if sv is not None:
            found[key] = sv
    return TerritoryAssetInputs(**found)
