"""Harvest the global deal universe into a normalized corpus.

A :class:`DealsSource` yields :class:`DealRecord`s from some backend. The live
sources — SEC EDGAR primary filings and commercial deal databases — are
config-gated stubs (wiring points). ``harvest`` merges all available sources,
dedupes, and persists to a JSON corpus. ``FileDealsSource`` reads an existing
corpus (e.g. the seed) so the analyzer and golden test run offline.

The seed corpus shipped in ``data/comparables/deals.json`` is **illustrative and
clearly labelled** (confidence ``assumption``, no source) — it exercises the
analyzer without masquerading as real disclosed terms. Replace it by running a
real harvest.
"""

from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from valuation_engine.comparables.schema import DealRecord


class DealsSource(abc.ABC):
    name: str = "deals_source"

    def available(self) -> bool:
        return True

    @abc.abstractmethod
    def fetch(self) -> list[DealRecord]:
        ...


class FileDealsSource(DealsSource):
    """Load a corpus from a JSON file (list of DealRecord mappings)."""

    name = "file"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def available(self) -> bool:
        return self.path.exists()

    def fetch(self) -> list[DealRecord]:
        if not self.available():
            return []
        raw = json.loads(self.path.read_text())
        return [DealRecord.model_validate(r) for r in raw]


# The real SEC EDGAR deals source lives in
# ``valuation_engine.research.data_sources.edgar.SecEdgarDealsSource`` (it depends
# on the EDGAR client). Import it from there; it implements this DealsSource API.


class CommercialDbDealsSource(DealsSource):
    """Commercial deal DB (Cortellis/DealForma/Evaluate). Config-gated stub."""

    name = "commercial_db"

    def available(self) -> bool:
        return bool(os.getenv("DEAL_DB_API_KEY"))

    def fetch(self) -> list[DealRecord]:  # pragma: no cover
        if not self.available():
            return []
        raise NotImplementedError("Wire commercial deal database API here.")


@dataclass
class HarvestResult:
    records: list[DealRecord]
    per_source: dict[str, int] = field(default_factory=dict)
    duplicates_dropped: int = 0


def _dedupe_key(r: DealRecord) -> tuple:
    # Prefer explicit id; else a natural key of the deal.
    return (r.licensor, r.licensee, r.asset, r.date, tuple(sorted(r.regions)))


def harvest(sources: Iterable[DealsSource]) -> HarvestResult:
    """Merge all available sources into a deduped corpus."""
    seen: dict[tuple, DealRecord] = {}
    per_source: dict[str, int] = {}
    dropped = 0
    for src in sources:
        if not src.available():
            continue
        recs = src.fetch()
        per_source[src.name] = len(recs)
        for r in recs:
            key = (r.id,) if r.id else _dedupe_key(r)
            if key in seen:
                dropped += 1
                continue
            seen[key] = r
    return HarvestResult(records=list(seen.values()), per_source=per_source, duplicates_dropped=dropped)


def save_corpus(records: list[DealRecord], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([r.model_dump(exclude_none=True) for r in records], indent=2)
    )


def load_corpus(path: str | Path) -> list[DealRecord]:
    return FileDealsSource(path).fetch()


def default_corpus_path() -> Path:
    """Repo-relative path to the shipped seed corpus."""
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "comparables" / "deals.json"
