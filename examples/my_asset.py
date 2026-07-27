"""TEMPLATE — value one drug across the three markets.

Copy this file, edit the numbers marked  <-- EDIT , and run:

    ./.venv/bin/python examples/my_asset.py           # after `pip install -e .`
    # or, without installing:
    PYTHONPATH=src python examples/my_asset.py

Every input is a number OR a SourcedValue. Use a bare number when you're sure
(`0.6`); use SourcedValue(value=.., low=.., high=..) to give a range that the
Monte Carlo will sample and the tornado will test:

    diagnosis_rate=SourcedValue(value=0.6, low=0.45, high=0.75,
        provenance={"source": "https://...", "confidence": "medium"})
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

# ---------------------------------------------------------------------------
# 1) THE DRUG  (asset-level facts — true everywhere)
# ---------------------------------------------------------------------------
asset = AssetSpec(
    id="MY-001",
    name="My Drug",
    modality_id="biologic",           # <-- EDIT: "small_molecule" | "biologic" | "gene_therapy"
    indication="Rheumatoid arthritis",  # <-- EDIT
    phase="phase3",                   # <-- EDIT: preclinical|phase1|phase2|phase3|filed|approved

    # Epidemiology funnel: population x rate x diagnosed x treated x eligible.
    epidemiology=EpidemiologyBlock(
        basis="prevalence",           # <-- EDIT: "prevalence" (chronic) or "incidence"
        rate_per_100k=SourcedValue(value=500, low=350, high=650),  # <-- EDIT per 100k
        diagnosis_rate=0.6,           # <-- EDIT fraction diagnosed
        treatment_rate=0.5,           # <-- EDIT fraction of diagnosed who get drug therapy
        eligible_fraction=0.2,        # <-- EDIT fraction eligible for THIS drug/line
    ),

    # Price: annual net-equivalent (recurring) OR per-course (gene therapy), USD.
    global_list_price_usd=SourcedValue(value=20000, low=15000, high=25000),  # <-- EDIT
    exclusivity_years=10,             # <-- EDIT years of on-market exclusivity
    peak_share=SourcedValue(value=0.25, low=0.15, high=0.35),  # <-- EDIT peak market share
    differentiation_score=1.1,        # <-- EDIT 1.0=parity vs SoC, >1 better, <1 worse

    # Optional: clinical LoA is injected by DefaultLoAProvider by phase/modality.
    # To override with your own number: clinical_loa_override=SourcedValue(value=0.7, kind="bernoulli"),

    # Optional: drug-specific, per-country facts (from your research). Anything
    # set here overrides the territory-pack default for that country.
    territory_overrides={
        "brazil": TerritoryAssetInputs(
            # p_reimbursement=SourcedValue(value=0.65, kind="bernoulli"),   # <-- e.g. HTA research
            # net_price_usd=SourcedValue(value=14000),                      # <-- direct net price
            competitors=[
                CompetitorEntry(
                    name="RivalDrug",
                    launch_year_from_now=SourcedValue(value=4, low=3, high=6),  # <-- EDIT
                    share_capture=SourcedValue(value=0.2, low=0.1, high=0.35),
                )
            ],
        ),
    },
)

# ---------------------------------------------------------------------------
# 2) THE DEAL  (how the territory value is split)
# ---------------------------------------------------------------------------
deal = DealTerms(
    deal_type="out_license",                       # <-- EDIT
    upfront_usd=SourcedValue(value=3_000_000),     # <-- EDIT
    milestones=[
        Milestone(label="Approval", amount_usd=5_000_000, trigger="on_approval"),      # <-- EDIT
        Milestone(label="Reimbursement", amount_usd=4_000_000, trigger="on_reimbursement"),
    ],
    royalty_tiers=[                                 # <-- EDIT tiered royalty on net sales
        RoyaltyTier(min_sales_usd=0, max_sales_usd=50_000_000, rate=0.10),
        RoyaltyTier(min_sales_usd=50_000_000, rate=0.15),
    ],
)

# ---------------------------------------------------------------------------
# 3) RUN across the three markets
# ---------------------------------------------------------------------------
def main() -> None:
    modality = load_modality(asset.modality_id)
    analyzer = ComparablesAnalyzer(load_corpus(default_corpus_path()))
    results = []
    for tid in ["mexico", "brazil", "saudi_arabia"]:   # <-- EDIT the market list
        tv = value_territory(
            asset=asset,
            modality=modality,
            territory=load_territory(tid),
            clinical_provider=DefaultLoAProvider(),
            deal=deal,
            analyzer=analyzer,
            n_mc=10_000,       # <-- EDIT Monte Carlo draws
            seed=2024,
        )
        results.append(tv)
        print(render_text(tv, show_ledger=(tid == "mexico")))
        print()
    print(render_portfolio(results))


if __name__ == "__main__":
    main()
