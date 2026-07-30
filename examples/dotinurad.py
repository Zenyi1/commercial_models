"""Value DOTINURAD (SURI, gout/hyperuricemia) across Mexico, Brazil, Saudi Arabia.

Runs with sourced research when VALYU_API_KEY is set, otherwise falls back to the
hand-entered assumptions below (all tagged confidence="assumption").

    ./.venv/bin/python examples/dotinurad.py
    VALYU_API_KEY=... ./.venv/bin/python examples/dotinurad.py   # live research

Every drug-specific number here is an ASSUMPTION until research (or your own
figures) replace it. With a Valyu key the engine overrides the researchable keys
(epidemiology funnel, reimbursement, timing, price, market-access spend) with
sourced values carrying a URL + quote; the deal terms are never researched.
"""
from __future__ import annotations

import os

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
from valuation_engine.research.cache import CachingProvider, ResearchCache
from valuation_engine.research.llm_extractor import make_llm_extractor
from valuation_engine.research.provider import EscalatingProvider, MultiSourceProvider
from valuation_engine.research.valyu_deepresearch import BatchDeepResearchProvider
from valuation_engine.research.valyu_provider import ValyuProvider


def sv(value, low=None, high=None, kind="point", note="Claude assumption"):
    return SourcedValue(
        value=value, low=low, high=high,
        kind=("pert" if low is not None else kind),
        provenance={"confidence": "assumption", "notes": note},
    )


asset = AssetSpec(
    id="DOTINURAD-001",
    name="Dotinurad (SURI)",
    modality_id="small_molecule",
    indication="gout",                     # drives the research queries
    phase="approved",                      # approved in Japan -> low clinical risk
    epidemiology=EpidemiologyBlock(
        basis="prevalence",
        rate_per_100k=sv(2000, 1200, 3500),
        diagnosis_rate=sv(0.5, 0.35, 0.65),
        treatment_rate=sv(0.40, 0.25, 0.55),
        eligible_fraction=sv(0.15, 0.08, 0.25),
    ),
    global_list_price_usd=sv(500, 300, 900),
    exclusivity_years=sv(8, 6, 10),
    peak_share=sv(0.20, 0.10, 0.30),
    differentiation_score=sv(1.1, 0.95, 1.25),
    territory_overrides={
        tid: TerritoryAssetInputs(
            competitors=[CompetitorEntry(
                name="Generic allopurinol/febuxostat", available_now=True,
                share_capture=sv(0.25, 0.15, 0.40),
            )],
        )
        for tid in ("mexico", "brazil", "saudi_arabia")
    },
)

deal = DealTerms(
    deal_type="out_license",
    upfront_usd=sv(2_000_000),
    milestones=[
        Milestone(label="Approval", amount_usd=sv(3_000_000), trigger="on_approval"),
        Milestone(label="Reimbursement", amount_usd=sv(2_000_000), trigger="on_reimbursement"),
    ],
    royalty_tiers=[
        RoyaltyTier(min_sales_usd=0, max_sales_usd=25_000_000, rate=0.08),
        RoyaltyTier(min_sales_usd=25_000_000, rate=0.12),
    ],
)


def build_research_provider():
    """Valyu when keyed; None otherwise (engine then uses the assumptions).

    If ANTHROPIC_API_KEY is also set, Valyu's passages are read by an LLM
    extractor (accurate on real prose) instead of the regex heuristic.
    """
    extractor = None
    mode = "regex extraction"
    if os.getenv("ANTHROPIC_API_KEY"):
        extractor = make_llm_extractor()
        mode = f"LLM extraction ({os.getenv('ANTHROPIC_MODEL', 'claude-opus-4-8')})"
    valyu = ValyuProvider(extractor=extractor)
    if not valyu.available():
        return None, "none — using assumptions"
    # A persistent cache/journal so paid research is paid for at most once per
    # question, and in-flight DeepResearch tasks are resumed, never orphaned.
    cache = ResearchCache(os.getenv("RESEARCH_CACHE_PATH", "data/research_cache.json"))
    refresh = os.getenv("RESEARCH_REFRESH") == "1"  # force re-fetch (still writes)

    # ENRICH_ESCALATE=1 adds a DeepResearch fallback for keys fast search can't
    # answer (slower/costlier — fires only on the residual None keys). Both tiers
    # are cache-backed, so a rerun collects prior results for free.
    if os.getenv("ENRICH_ESCALATE") == "1":
        # One combined DeepResearch task per territory, Haiku extracts each field
        # off the shared report. RESEARCH_ALLOW_ESTIMATE lets it derive a grounded
        # low-confidence figure when none is stated outright.
        allow_estimate = os.getenv("RESEARCH_ALLOW_ESTIMATE") == "1"
        deep_extractor = (make_llm_extractor(max_chars=24000, allow_estimate=allow_estimate)
                          if os.getenv("ANTHROPIC_API_KEY") else None)
        deep = BatchDeepResearchProvider(extractor=deep_extractor, cache=cache, refresh=refresh)
        ladder = EscalatingProvider([CachingProvider(valyu, cache, refresh), deep])
        est = " +estimate" if allow_estimate else ""
        return MultiSourceProvider([ladder]), f"Valyu (live) + {mode} + DeepResearch(batch){est} [cached]"
    return MultiSourceProvider([CachingProvider(valyu, cache, refresh)]), f"Valyu (live) + {mode} [cached]"


def main() -> None:
    modality = load_modality(asset.modality_id)
    analyzer = ComparablesAnalyzer(load_corpus(default_corpus_path()))
    research, mode = build_research_provider()
    print("Research:", mode)
    results = []
    for tid in ["mexico", "brazil", "saudi_arabia"]:
        tv = value_territory(
            asset=asset,
            modality=modality,
            territory=load_territory(tid),
            clinical_provider=DefaultLoAProvider(),
            research_provider=research,
            deal=deal,
            analyzer=analyzer,
            n_mc=10_000,
            seed=2024,
        )
        results.append(tv)
        print(render_text(tv, show_ledger=(tid == "mexico")))
        print()
    print(render_portfolio(results))


if __name__ == "__main__":
    main()
