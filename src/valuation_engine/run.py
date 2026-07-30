"""Value any asset across territories from data — no code edits per run.

    python -m valuation_engine.run --asset dotinurad --territories mexico,brazil,saudi_arabia
    python -m valuation_engine.run --asset dotinurad --research      # enable Valyu/LLM enrichment

The asset, deal, territories, and modalities are all JSON packs. Adding a drug =
adding ``assets/<id>.json`` (and optionally ``deals/<id>.json``); the engine and
this runner never change.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from valuation_engine.clinical.provider import DefaultLoAProvider
from valuation_engine.comparables.analyzer import ComparablesAnalyzer
from valuation_engine.comparables.harvest import default_corpus_path, load_corpus
from valuation_engine.inputs.packs import (
    available_assets,
    load_asset,
    load_deal,
    load_modality,
    load_territory,
)
from valuation_engine.report import render_portfolio, render_text, value_territory
from valuation_engine.research.provider import ResearchProvider


def _research_provider(enable: bool) -> Optional[ResearchProvider]:
    """Build the research ladder only if --research and a Valyu key are present.

    Imported lazily so the default (no-research) path has zero research deps.
    """
    if not enable:
        return None
    from valuation_engine.research.cache import CachingProvider, ResearchCache
    from valuation_engine.research.llm_extractor import make_llm_extractor
    from valuation_engine.research.provider import EscalatingProvider, MultiSourceProvider
    from valuation_engine.research.valyu_deepresearch import BatchDeepResearchProvider
    from valuation_engine.research.valyu_provider import ValyuProvider

    extractor = make_llm_extractor() if os.getenv("ANTHROPIC_API_KEY") else None
    valyu = ValyuProvider(extractor=extractor)
    if not valyu.available():
        return None
    cache = ResearchCache(os.getenv("RESEARCH_CACHE_PATH", "data/research_cache.json"))
    fast = CachingProvider(valyu, cache)
    if os.getenv("ENRICH_ESCALATE") == "1":
        deep = BatchDeepResearchProvider(extractor=extractor, cache=cache)
        return MultiSourceProvider([EscalatingProvider([fast, deep])])
    return MultiSourceProvider([fast])


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Value an asset across territories (data-driven).")
    ap.add_argument("--asset", required=True, help=f"asset id (available: {', '.join(available_assets()) or 'none'})")
    ap.add_argument("--territories", default="mexico,brazil,saudi_arabia",
                    help="comma-separated territory ids")
    ap.add_argument("--deal", default=None, help="deal id (default: same id as --asset if it exists)")
    ap.add_argument("--research", action="store_true", help="enable Valyu/LLM enrichment (needs VALYU key)")
    ap.add_argument("--n-mc", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args(argv)

    asset = load_asset(args.asset)
    modality = load_modality(asset.modality_id)
    try:
        deal = load_deal(args.deal or args.asset)
    except FileNotFoundError:
        deal = None
    analyzer = ComparablesAnalyzer(load_corpus(default_corpus_path()))
    research = _research_provider(args.research)

    print(f"Asset: {asset.name} ({asset.id}) | modality {asset.modality_id} | "
          f"deal {'yes' if deal else 'none'} | research {'on' if research else 'off'}")
    results = []
    territories = [t.strip() for t in args.territories.split(",") if t.strip()]
    for i, tid in enumerate(territories):
        tv = value_territory(
            asset=asset, modality=modality, territory=load_territory(tid),
            clinical_provider=DefaultLoAProvider(), research_provider=research,
            deal=deal, analyzer=analyzer, n_mc=args.n_mc, seed=args.seed,
        )
        results.append(tv)
        print(render_text(tv, show_ledger=(i == 0)))
        print()
    print(render_portfolio(results))


if __name__ == "__main__":
    main()
