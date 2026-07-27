"""Harvest real comparable deals from SEC EDGAR into a corpus file.

Usage:
    python scripts/harvest_edgar.py \
        --user-agent "FirstOcean Research you@example.com" \
        --start 2023-01-01 --end 2023-12-31 --limit 100 \
        --out data/comparables/deals_edgar.json

SEC requires a User-Agent with a contact email (or set SEC_EDGAR_USER_AGENT).
By default this writes to ``deals_edgar.json`` so it does not clobber the seed
``deals.json``; point the engine at it (or replace the seed) once you've spot-
checked the extracted terms against the filings — every record carries its
source URL. Royalty rates are often redacted in filings and will be null; that
is expected and honest, not a bug.
"""

from __future__ import annotations

import argparse

from valuation_engine.comparables.harvest import harvest, save_corpus
from valuation_engine.research.data_sources.edgar import DEFAULT_QUERY, SecEdgarDealsSource


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest comparable deals from SEC EDGAR")
    ap.add_argument("--user-agent", default=None, help="SEC UA, e.g. 'Org Name you@example.com' (or SEC_EDGAR_USER_AGENT)")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--forms", default="8-K")
    ap.add_argument("--start", default="", help="YYYY-MM-DD")
    ap.add_argument("--end", default="", help="YYYY-MM-DD")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-fetch", action="store_true", help="list filings without extracting terms (fast)")
    ap.add_argument("--out", default="data/comparables/deals_edgar.json")
    args = ap.parse_args()

    source = SecEdgarDealsSource(
        user_agent=args.user_agent, query=args.query, forms=args.forms,
        startdt=args.start, enddt=args.end, limit=args.limit,
        fetch_documents=not args.no_fetch,
    )
    if not source.available():
        raise SystemExit("No SEC User-Agent. Pass --user-agent or set SEC_EDGAR_USER_AGENT (must include a contact email).")

    print(f"Searching EDGAR: q={args.query!r} forms={args.forms} {args.start}..{args.end} limit={args.limit}")
    result = harvest([source])
    save_corpus(result.records, args.out)

    with_terms = sum(1 for r in result.records if r.upfront_usd is not None or r.peak_royalty_rate is not None)
    with_royalty = sum(1 for r in result.records if r.peak_royalty_rate is not None)
    print(f"Harvested {len(result.records)} filings ({result.duplicates_dropped} dupes dropped)")
    print(f"  with extractable terms: {with_terms}   with royalty rate: {with_royalty}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
