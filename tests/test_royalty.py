"""Tests for royalty imputation."""

import pytest

from valuation_engine.comparables.royalty import INDUSTRY_STAGE_PRIOR, RoyaltyImputer
from valuation_engine.comparables.schema import DealRecord


def _rec(i, stage="phase3", modality="biologic", royalty=None):
    return DealRecord(id=str(i), stage=stage, modality=modality, peak_royalty_rate=royalty)


def test_impute_from_stratum_median():
    recs = [_rec(i, royalty=r) for i, r in enumerate([0.10, 0.12, 0.14, 0.16])]
    imp = RoyaltyImputer(recs, min_n=3)
    sv = imp.impute("phase3", "biologic")
    assert sv.low <= sv.value <= sv.high
    assert sv.value == pytest.approx(0.13, abs=0.01)  # median of the four
    assert sv.provenance.confidence == "low"
    assert "IMPUTED" in (sv.provenance.notes or "")


def test_hierarchical_fallback_to_broader_stratum():
    # Only 1 phase3/biologic (too thin), but many phase3 across modalities.
    recs = [_rec(0, "phase3", "biologic", 0.20)]
    recs += [_rec(i, "phase3", "small_molecule", r) for i, r in enumerate([0.10, 0.11, 0.12], start=1)]
    imp = RoyaltyImputer(recs, min_n=3)
    sv = imp.impute("phase3", "biologic")
    # Falls back to stage=phase3 across modalities -> median ~0.115, not 0.20.
    assert sv.value < 0.15
    assert "stage=phase3" in (sv.provenance.method or "")


def test_falls_back_to_industry_prior_when_no_data():
    imp = RoyaltyImputer([], min_n=3)
    sv = imp.impute("phase3", "biologic")
    lo, base, hi = INDUSTRY_STAGE_PRIOR["phase3"]
    assert sv.value == base and sv.low == lo and sv.high == hi
    assert sv.provenance.confidence == "assumption"
    assert "rule-of-thumb" in (sv.provenance.notes or "")


def test_industry_prior_escalates_with_stage():
    imp = RoyaltyImputer([])
    assert imp.impute("phase3").value > imp.impute("phase1").value


def test_augment_fills_missing_only_and_flags():
    recs = [_rec(0, royalty=0.15), _rec(1, royalty=None), _rec(2, royalty=None)]
    # give the imputer enough disclosed data via a separate calibration corpus
    calib = [_rec(i, royalty=r) for i, r in enumerate([0.10, 0.12, 0.14], start=10)]
    imp = RoyaltyImputer(recs + calib, min_n=3)
    out = imp.augment(recs)
    assert out[0].peak_royalty_rate == 0.15  # disclosed untouched
    assert out[0].confidence == recs[0].confidence
    assert out[1].peak_royalty_rate is not None  # imputed
    assert out[1].confidence == "low"
    assert "imputed" in (out[1].notes or "").lower()


def test_coverage_reports_disclosed_count():
    recs = [_rec(0, royalty=0.1), _rec(1, royalty=None), _rec(2, royalty=0.2)]
    assert RoyaltyImputer(recs).coverage()["disclosed_rates"] == 2
