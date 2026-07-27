"""Normalized comparable-deal record.

One row per historical licensing / territorial / M&A deal, harvested from
primary filings (SEC EDGAR) and commercial deal databases. Fields are optional
because disclosure is uneven; the analyzer filters to whatever facets a query
needs. Every record carries a source and a confidence so the corpus stays
auditable.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Modality = Literal["small_molecule", "biologic", "gene_therapy", "other"]
Stage = Literal["preclinical", "phase1", "phase2", "phase3", "filed", "approved", "unknown"]
DealType = Literal[
    "global_license", "regional_license", "ex_us_license", "distribution", "mna", "other"
]

# Coarse region tags a deal's rights can cover.
Region = Literal["global", "us", "ex_us", "eu", "latam", "mena", "apac", "china", "japan", "row"]


class DealRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    asset: Optional[str] = None
    licensor: Optional[str] = None
    licensee: Optional[str] = None
    modality: Modality = "other"
    stage: Stage = "unknown"
    indication: Optional[str] = None
    deal_type: DealType = "other"
    regions: list[Region] = Field(default_factory=lambda: ["global"])
    date: Optional[str] = None  # ISO year or date

    upfront_usd: Optional[float] = None
    total_value_usd: Optional[float] = None  # incl milestones ("biobucks")
    peak_royalty_rate: Optional[float] = None  # top tier, 0..1
    disclosed_peak_sales_usd: Optional[float] = None

    source: Optional[str] = None
    confidence: Literal["high", "medium", "low", "assumption"] = "medium"
    notes: Optional[str] = None
