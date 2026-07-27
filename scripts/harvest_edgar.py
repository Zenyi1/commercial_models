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

from valuation_engine.comparables.harvest import harvest, load_corpus, save_corpus
from valuation_engine.comparables.royalty import RoyaltyImputer
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
    ap.add_argument("--impute-royalties", action="store_true",
                    help="fill redacted royalty rates from disclosed comps (flagged as imputed)")
    ap.add_argument("--calibrate-from", default=None,
                    help="extra corpus JSON to calibrate royalty imputation (adds its disclosed rates)")
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
    records = result.records

    disclosed = sum(1 for r in records if r.peak_royalty_rate is not None)
    if args.impute_royalties:
        calibration = list(records)
        if args.calibrate_from:
            calibration += load_corpus(args.calibrate_from)
        imputer = RoyaltyImputer(calibration)
        print(f"Imputing royalties (calibrated on {imputer.coverage()['disclosed_rates']} disclosed rates; "
              "industry prior where sparse). Imputed rows are flagged, not presented as disclosed.")
        records = imputer.augment(records)

    save_corpus(records, args.out)

    with_terms = sum(1 for r in records if r.upfront_usd is not None or r.peak_royalty_rate is not None)
    with_royalty = sum(1 for r in records if r.peak_royalty_rate is not None)
    print(f"Harvested {len(records)} filings ({result.duplicates_dropped} dupes dropped)")
    print(f"  with extractable terms: {with_terms}   royalty disclosed: {disclosed}   royalty total (incl imputed): {with_royalty}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
