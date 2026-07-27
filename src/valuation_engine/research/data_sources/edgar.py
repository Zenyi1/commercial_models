"""SEC EDGAR deals source — real comparable deals from primary filings.

Two layers, kept separate so the parsing/extraction logic is testable without
the network:

* :class:`EdgarClient` — thin HTTP wrapper over EDGAR full-text search (EFTS) and
  document retrieval. Sends the SEC-required ``User-Agent`` and self-rate-limits.
* pure functions — :func:`parse_search_results`, :func:`extract_deal_terms`,
  :func:`classify_modality` / :func:`classify_stage`, :func:`extract_parties` —
  turn EDGAR JSON/text into structured fields with no I/O.

:class:`SecEdgarDealsSource` (below) ties them into a
:class:`~valuation_engine.comparables.harvest.DealsSource`. Every record it
produces carries the filing URL as its source; financial fields are populated
only where the text yields them with confidence, and left ``None`` otherwise —
never guessed.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
_MAX_PAGE = 100  # EFTS returns at most 100 hits per request


@dataclass(frozen=True)
class EdgarHit:
    accession: str          # e.g. 0001193125-23-002398
    filename: str           # primary document filename
    cik: str                # zero-padded CIK of the filer
    filer_name: str         # display name of the filer
    form: str               # 8-K, 10-K, ...
    file_date: str          # ISO date

    @property
    def document_url(self) -> str:
        cik_int = str(int(self.cik))  # strip leading zeros
        acc_nodash = self.accession.replace("-", "")
        return f"{ARCHIVES_URL}/{cik_int}/{acc_nodash}/{self.filename}"

    @property
    def filing_index_url(self) -> str:
        cik_int = str(int(self.cik))
        acc_nodash = self.accession.replace("-", "")
        return f"{ARCHIVES_URL}/{cik_int}/{acc_nodash}/"


def _clean_filer_name(display_name: str) -> str:
    """'CytomX Therapeutics, Inc.  (CTMX)  (CIK 0001501989)' -> 'CytomX Therapeutics, Inc.'"""
    return display_name.split("  (")[0].strip()


def parse_search_results(payload: dict) -> list[EdgarHit]:
    """Parse an EFTS JSON response into :class:`EdgarHit`s (pure, no I/O)."""
    hits: list[EdgarHit] = []
    for h in payload.get("hits", {}).get("hits", []):
        _id = h.get("_id", "")
        if ":" not in _id:
            continue
        accession, filename = _id.split(":", 1)
        src = h.get("_source", {})
        ciks = src.get("ciks") or [""]
        names = src.get("display_names") or [""]
        hits.append(EdgarHit(
            accession=accession,
            filename=filename,
            cik=ciks[0],
            filer_name=_clean_filer_name(names[0]),
            form=src.get("form", ""),
            file_date=src.get("file_date", ""),
        ))
    return hits


def total_hits(payload: dict) -> int:
    return int(payload.get("hits", {}).get("total", {}).get("value", 0))


class EdgarClient:
    """Minimal EDGAR client. ``user_agent`` MUST identify you per SEC policy,
    e.g. ``"FirstOcean Research you@example.com"``."""

    def __init__(self, user_agent: str, min_interval_s: float = 0.15):
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with contact email, e.g. 'Org Name you@example.com'")
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()

    def _get(self, url: str, params: Optional[dict] = None) -> bytes:
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        self._throttle()
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/html",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
        return raw

    def search_page(
        self, query: str, forms: str = "8-K", startdt: str = "", enddt: str = "",
        from_offset: int = 0,
    ) -> dict:
        params = {"q": query, "forms": forms, "from": from_offset}
        if startdt:
            params["startdt"] = startdt
        if enddt:
            params["enddt"] = enddt
        return json.loads(self._get(EFTS_URL, params))

    def search(
        self, query: str, forms: str = "8-K", startdt: str = "", enddt: str = "",
        limit: int = 100,
    ) -> list[EdgarHit]:
        """Paginate EFTS up to ``limit`` hits."""
        out: list[EdgarHit] = []
        offset = 0
        while len(out) < limit:
            payload = self.search_page(query, forms, startdt, enddt, offset)
            page = parse_search_results(payload)
            if not page:
                break
            out.extend(page)
            offset += _MAX_PAGE
            if offset >= total_hits(payload):
                break
        return out[:limit]

    def fetch_document_text(self, hit: EdgarHit) -> str:
        """Fetch a filing's primary document and strip HTML to plain text."""
        raw = self._get(hit.document_url).decode("utf-8", errors="ignore")
        return _html_to_text(raw)


def _html_to_text(html: str) -> str:
    import re
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#160;", " ").replace("&#8217;", "'").replace("&rsquo;", "'"))
    text = re.sub(r"\s+", " ", text)
    return text.strip()
