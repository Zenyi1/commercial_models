"""End-to-end example: value the territorial rights to one asset across
Mexico, Brazil and Saudi Arabia.

Run:  PYTHONPATH=src python examples/value_asset.py

This is the reference driver. It builds a Phase 3 biologic, loads the sourced
territory/modality packs, harvests the (seed) comparables corpus, and prints a
full per-territory valuation plus a portfolio summary. Swap in a live
ResearchProvider / ClinicalModelProvider to replace the defaults.
"""

from __future__ import annotations

from valuation_engine.clinical.provider import DefaultLoAProvider
from valuation_engine.comparables.analyzer import ComparablesAnalyzer
from valuation_engine.comparables.harvest import default_corpus_path, load_corpus
from valuation_engine.inputs.packs import load_modality, load_territory
from valuation_engine.inputs.schemas import (
    AssetSpec,
    CompetitorEntry,
    DealTerms,
    EpidemiologyBlock,
    Milestone,
    RoyaltyTier,
    SourcedValue,
    TerritoryAssetInputs,
)
from valuation_engine.report import render_portfolio, render_text, value_territory


def build_asset() -> AssetSpec:
    return AssetSpec(
        id="FO-001",
        name="FO-001 (anti-IL-X mAb)",
        modality_id="biologic",
        indication="Rheumatoid arthritis",
        phase="phase3",
        epidemiology=EpidemiologyBlock(
            basis="prevalence",
            rate_per_100k=SourcedValue(value=500, low=350, high=650,
                provenance={"method": "RA prevalence per 100k", "confidence": "low"}),
            diagnosis_rate=SourcedValue(value=0.6, low=0.45, high=0.75),
            treatment_rate=SourcedValue(value=0.5, low=0.35, high=0.65),
            eligible_fraction=SourcedValue(value=0.2, low=0.12, high=0.30,
                provenance={"method": "Share eligible for this MoA/line", "confidence": "low"}),
        ),
        global_list_price_usd=SourcedValue(value=20000, low=15000, high=25000,
            provenance={"method": "Annual net-equivalent global anchor", "confidence": "low"}),
        exclusivity_years=SourcedValue(value=10, low=8, high=12),
        peak_share=SourcedValue(value=0.25, low=0.15, high=0.35),
        uptake_years_to_peak=SourcedValue(value=5, low=3, high=7),
        differentiation_score=SourcedValue(value=1.1, low=0.9, high=1.3,
            provenance={"method": "Modest edge vs in-market SoC", "confidence": "assumption"}),
        # Example asset x territory override: a competitor arriving in Brazil in yr 4.
        territory_overrides={
            "brazil": TerritoryAssetInputs(
                competitors=[CompetitorEntry(
                    name="RivalBio", launch_year_from_now=SourcedValue(value=4, low=3, high=6),
                    share_capture=SourcedValue(value=0.2, low=0.1, high=0.35))]
            )
        },
    )


def build_deal() -> DealTerms:
    return DealTerms(
        deal_type="out_license",
        upfront_usd=SourcedValue(value=3_000_000, low=1_000_000, high=6_000_000),
        milestones=[
            Milestone(label="Regulatory approval", amount_usd=SourcedValue(value=5_000_000, low=3e6, high=8e6),
                      trigger="on_approval"),
            Milestone(label="Reimbursement listing", amount_usd=4_000_000, trigger="on_reimbursement"),
            Milestone(label="Sales > $50M", amount_usd=10_000_000, trigger="on_sales",
                      sales_threshold_usd=50_000_000),
        ],
        royalty_tiers=[
            RoyaltyTier(min_sales_usd=0, max_sales_usd=50_000_000, rate=0.10),
            RoyaltyTier(min_sales_usd=50_000_000, rate=0.15),
        ],
    )


def main() -> None:
    asset = build_asset()
    modality = load_modality("biologic")
    deal = build_deal()
    clinical = DefaultLoAProvider()
    analyzer = ComparablesAnalyzer(load_corpus(default_corpus_path()))

    valuations = []
    for tid in ["mexico", "brazil", "saudi_arabia"]:
        territory = load_territory(tid)
        tv = value_territory(
            asset=asset, modality=modality, territory=territory,
            clinical_provider=clinical, deal=deal, analyzer=analyzer,
            # reference_value_usd would be the asset's US/global rNPV from your own
            # model; supplying it adds a geography-adjusted comparables leg. Omitted
            # here so triangulation shows the two internally-derived legs.
            n_mc=8000, seed=2024,
        )
        valuations.append(tv)
        print(render_text(tv, show_ledger=(tid == "mexico")))
        print()

    print(render_portfolio(valuations))


if __name__ == "__main__":
    main()
