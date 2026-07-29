"""Enrichment fans out per-key research calls concurrently and collects only
the answered keys. No network — a fake provider stands in."""

import threading
import time

from valuation_engine.inputs.schemas import (
    AssetSpec,
    EpidemiologyBlock,
    SourcedValue,
)
from valuation_engine.research.enrich import enrich_territory_asset
from valuation_engine.research.provider import ResearchProvider


def _asset():
    return AssetSpec(
        id="A1",
        name="Test Asset",
        modality_id="small_molecule",
        indication="gout",
        phase="approved",
        epidemiology=EpidemiologyBlock(
            basis="prevalence", rate_per_100k=1000, diagnosis_rate=0.5,
            treatment_rate=0.4,
        ),
        global_list_price_usd=500,
        exclusivity_years=8,
    )


class _SlowProvider(ResearchProvider):
    """Answers a fixed set of keys after a small delay; tracks peak concurrency."""

    name = "slow"

    def __init__(self, answer_keys, delay=0.05):
        self.answer_keys = set(answer_keys)
        self.delay = delay
        self._live = 0
        self.peak = 0
        self._lock = threading.Lock()

    def get(self, query):
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)
        try:
            time.sleep(self.delay)
            if query.key in self.answer_keys:
                return SourcedValue(value=0.5, kind="bernoulli")
            return None
        finally:
            with self._lock:
                self._live -= 1


def test_enrich_collects_only_answered_keys():
    prov = _SlowProvider(answer_keys={"p_reimbursement", "treatment_rate"}, delay=0.0)
    out = enrich_territory_asset(prov, _asset(), "brazil")
    assert out.p_reimbursement is not None
    assert out.treatment_rate is not None
    assert out.peak_share is None  # not answered -> stays None


def test_enrich_runs_concurrently():
    prov = _SlowProvider(answer_keys={"p_reimbursement"}, delay=0.05)
    t0 = time.monotonic()
    enrich_territory_asset(prov, _asset(), "brazil", max_workers=8)
    elapsed = time.monotonic() - t0
    # 13 keys x 0.05s = 0.65s sequential; concurrent should be far less.
    assert elapsed < 0.4, f"enrich did not parallelize (took {elapsed:.2f}s)"
    assert prov.peak > 1, f"no concurrency observed (peak={prov.peak})"


def test_enrich_empty_keys_returns_blank():
    prov = _SlowProvider(answer_keys=set())
    out = enrich_territory_asset(prov, _asset(), "brazil", keys=[])
    assert out.p_reimbursement is None
