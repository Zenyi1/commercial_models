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
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from valuation_engine.comparables.harvest import DealsSource
from valuation_engine.comparables.schema import DealRecord

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


# --------------------------------------------------------------------------- #
# Extraction (pure, testable) — populate a field only when confident, else None.
# --------------------------------------------------------------------------- #
import re  # noqa: E402

_UNIT = {"billion": 1e9, "million": 1e6, "thousand": 1e3}
_MONEY = r"\$\s?([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)?"


def _to_usd(num: str, unit: Optional[str]) -> float:
    return float(num.replace(",", "")) * _UNIT.get((unit or "").lower(), 1.0)


def _amount_near(text: str, keyword: str, window: int = 70) -> Optional[float]:
    """USD amount of the money token *closest* to ``keyword`` on either side.

    Money commonly precedes the keyword ("$170 million in upfront payments") as
    often as it follows it, so we search a symmetric window and pick the nearest
    match. A magnitude unit (million/billion/thousand) is required, which avoids
    grabbing stray "$5" fragments and share-price figures.
    """
    best: Optional[float] = None
    best_dist = 10**9
    for km in re.finditer(re.escape(keyword), text, re.I):
        kpos = (km.start() + km.end()) // 2
        lo, hi = max(0, kpos - window), kpos + window
        segment = text[lo:hi]
        for mm in re.finditer(_MONEY, segment):
            if not mm.group(2):  # require an explicit unit
                continue
            mpos = lo + (mm.start() + mm.end()) // 2
            dist = abs(mpos - kpos)
            if dist < best_dist:
                best_dist = dist
                best = _to_usd(mm.group(1), mm.group(2))
    return best


def extract_deal_terms(text: str) -> dict:
    """Extract upfront, total (biobucks) and peak royalty from filing text.

    Conservative: matches common disclosure phrasings and returns ``None`` for
    anything not found (royalty rates are frequently redacted). Values are USD.
    """
    out: dict[str, Optional[float]] = {
        "upfront_usd": None, "total_value_usd": None, "peak_royalty_rate": None,
    }

    out["upfront_usd"] = _amount_near(text, "upfront")

    # Milestones / biobucks: the amount near "milestone(s)".
    milestones = _amount_near(text, "milestone")
    if milestones is not None:
        # Total deal value = upfront + milestones when both known, else milestones.
        out["total_value_usd"] = (out["upfront_usd"] or 0.0) + milestones

    out["peak_royalty_rate"] = _extract_royalty(text)
    return out


# Plausible peak-tier royalty band. Rates above ~30% in filings are almost
# always profit-shares or equity, not royalties; below 0.5% are usually
# fragments. Values outside the band are rejected rather than trusted.
ROYALTY_MIN, ROYALTY_MAX = 0.005, 0.30

# Verbal royalty phrasings ("high single-digit royalties") -> representative
# peak rate. Lower-fidelity than a stated number, but common and better than
# discarding the signal.
_VERBAL_ROYALTY = [
    (r"high[\s-]*single[\s-]*digit", 0.085),
    (r"mid[\s-]*single[\s-]*digit", 0.055),
    (r"low[\s-]*single[\s-]*digit", 0.03),
    (r"single[\s-]*digit", 0.05),
    (r"high[\s-]*(?:double|teens?|ten)[\s-]*digit|high teens", 0.18),
    (r"mid[\s-]*double[\s-]*digit", 0.15),
    (r"low[\s-]*double[\s-]*digit|low teens", 0.12),
    (r"double[\s-]*digit|teens", 0.13),
]


def _extract_royalty(text: str) -> Optional[float]:
    # 1) numeric range near "royalt" -> take the top/peak, within plausibility band
    rng = re.search(
        r"(?i)royalt[^.%]{0,80}?(\d{1,2}(?:\.\d+)?)\s?%\s*(?:to|through|-|–|and)\s*(\d{1,2}(?:\.\d+)?)\s?%",
        text,
    )
    if rng:
        rate = float(rng.group(2)) / 100.0
        if ROYALTY_MIN <= rate <= ROYALTY_MAX:
            return rate
    # 2) a single numeric rate adjacent to "royalt"
    single = re.search(r"(?i)(\d{1,2}(?:\.\d+)?)\s?%\s*royalt|royalt[^.%]{0,60}?(\d{1,2}(?:\.\d+)?)\s?%", text)
    if single:
        rate = float(single.group(1) or single.group(2)) / 100.0
        if ROYALTY_MIN <= rate <= ROYALTY_MAX:
            return rate
    # 3) verbal band near "royalt"
    for m in re.finditer(r"(?i)royalt\w*", text):
        window = text[max(0, m.start() - 40): m.end() + 40]
        for pattern, value in _VERBAL_ROYALTY:
            if re.search("(?i)" + pattern, window):
                return value
    return None


def classify_modality(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("gene therapy", "aav", "gene-therapy", "cell therapy", "car-t", "car t")):
        return "gene_therapy"
    if any(k in t for k in ("monoclonal antibody", "antibody", "biologic", "mab", "fusion protein", "mrna", "protein")):
        return "biologic"
    if any(k in t for k in ("small molecule", "small-molecule", "oral", "inhibitor", "compound")):
        return "small_molecule"
    return "other"


def classify_stage(text: str) -> str:
    t = text.lower()
    for phrase, stage in (
        ("phase 3", "phase3"), ("phase iii", "phase3"),
        ("phase 2", "phase2"), ("phase ii", "phase2"),
        ("phase 1", "phase1"), ("phase i", "phase1"),
        ("preclinical", "preclinical"), ("pre-clinical", "preclinical"),
    ):
        if phrase in t:
            return stage
    if "fda-approved" in t or "fda approved" in t:
        return "approved"
    return "unknown"


def extract_regions(text: str) -> list[str]:
    t = text.lower()
    regions: list[str] = []
    if "worldwide" in t or "global" in t:
        regions.append("global")
    if "ex-u.s." in t or "ex-us" in t or "outside the united states" in t:
        regions.append("ex_us")
    if "latin america" in t:
        regions.append("latam")
    if "middle east" in t or "mena" in t:
        regions.append("mena")
    if "greater china" in t or ("china" in t and "china" != t):
        regions.append("china")
    if "japan" in t:
        regions.append("japan")
    return regions or ["global"]


def _deal_type_from_regions(regions: list[str]) -> str:
    if "global" in regions:
        return "global_license"
    if "ex_us" in regions:
        return "ex_us_license"
    return "regional_license"


# Default query: license/collaboration agreements likely to disclose economics.
DEFAULT_QUERY = '"license agreement" "upfront"'


class SecEdgarDealsSource(DealsSource):
    """Harvest real deals from EDGAR full-text search + filing extraction.

    Enabled when a SEC ``user_agent`` (with contact email) is provided, directly
    or via ``SEC_EDGAR_USER_AGENT``. Each hit becomes a :class:`DealRecord`
    sourced to the filing URL; term fields are filled only where extraction is
    confident. Set ``fetch_documents=False`` to list filings without pulling
    each document (fast, terms left None).
    """

    name = "sec_edgar"

    def __init__(
        self,
        user_agent: Optional[str] = None,
        query: str = DEFAULT_QUERY,
        forms: str = "8-K",
        startdt: str = "",
        enddt: str = "",
        limit: int = 50,
        fetch_documents: bool = True,
    ):
        self.user_agent = user_agent or os.getenv("SEC_EDGAR_USER_AGENT")
        self.query = query
        self.forms = forms
        self.startdt = startdt
        self.enddt = enddt
        self.limit = limit
        self.fetch_documents = fetch_documents

    def available(self) -> bool:
        return bool(self.user_agent)

    def _make_client(self) -> EdgarClient:
        return EdgarClient(self.user_agent)

    def _record_from_hit(self, client: EdgarClient, hit: EdgarHit) -> DealRecord:
        terms: dict = {"upfront_usd": None, "total_value_usd": None, "peak_royalty_rate": None}
        modality, stage, regions = "other", "unknown", ["global"]
        if self.fetch_documents:
            try:
                text = client.fetch_document_text(hit)
                terms = extract_deal_terms(text)
                modality = classify_modality(text)
                stage = classify_stage(text)
                regions = extract_regions(text)
            except Exception:  # a single bad document must not abort the harvest
                pass
        has_terms = terms["upfront_usd"] is not None or terms["peak_royalty_rate"] is not None
        return DealRecord(
            id=hit.accession,
            licensor=hit.filer_name,
            modality=modality,
            stage=stage,
            deal_type=_deal_type_from_regions(regions),
            regions=regions,
            date=hit.file_date,
            upfront_usd=terms["upfront_usd"],
            total_value_usd=terms["total_value_usd"],
            peak_royalty_rate=terms["peak_royalty_rate"],
            source=hit.document_url,
            confidence="medium" if has_terms else "low",
            notes=f"Auto-extracted from {hit.form} filed by {hit.filer_name}; verify terms against the filing.",
        )

    def fetch(self) -> list[DealRecord]:
        if not self.available():
            return []
        client = self._make_client()
        hits = client.search(self.query, self.forms, self.startdt, self.enddt, self.limit)
        return [self._record_from_hit(client, h) for h in hits]
