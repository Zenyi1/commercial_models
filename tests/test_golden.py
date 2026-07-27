"""Golden end-to-end snapshot.

Pins the full pipeline (packs -> resolve -> base rNPV -> deal split -> seeded
Monte Carlo -> decision) for a fixed Phase 3 biologic across Mexico, Brazil and
Saudi Arabia. The base case is deterministic and the Monte Carlo is seeded, so
these values are reproducible; any drift is a real behavioural change to
investigate. Regenerate deliberately if the model math is intentionally revised.
"""

import pytest

from valuation_engine.clinical.provider import DefaultLoAProvider
from valuation_engine.inputs.packs import load_modality, load_territory
from valuation_engine.inputs.schemas import (
    AssetSpec,
    DealTerms,
    EpidemiologyBlock,
    RoyaltyTier,
    SourcedValue,
)
from valuation_engine.report import value_territory

# (base_pie_rnpv, base_net_value, base_licensor_value, mc_p50_pie, mc_p_positive)
GOLDEN = {
    "mexico": (32_667_308.010496, 29_807_449.137494, 11_340_589.279275, 22_125_043.939396, 0.534333),
    "brazil": (39_822_337.323329, 35_578_958.723687, 13_167_405.274041, 21_013_666.485939, 0.529667),
    "saudi_arabia": (15_003_680.232687, 12_510_386.828880, 6_750_920.058172, 5_090_093.985382, 0.513667),
}


def _golden_asset() -> AssetSpec:
    return AssetSpec(
        id="GOLD", name="Gold", modality_id="biologic", indication="RA", phase="phase3",
        epidemiology=EpidemiologyBlock(
            basis="prevalence",
            rate_per_100k=SourcedValue(value=500, low=350, high=650),
            diagnosis_rate=0.6, treatment_rate=0.5,
            eligible_fraction=SourcedValue(value=0.2, low=0.12, high=0.3),
        ),
        global_list_price_usd=SourcedValue(value=20000, low=15000, high=25000),
        exclusivity_years=10, peak_share=SourcedValue(value=0.25, low=0.15, high=0.35),
    )


def _golden_deal() -> DealTerms:
    return DealTerms(deal_type="out_license", upfront_usd=3_000_000,
                     royalty_tiers=[RoyaltyTier(min_sales_usd=0, rate=0.12)])


@pytest.mark.parametrize("tid", sorted(GOLDEN))
def test_golden_snapshot(tid):
    asset, deal, mod = _golden_asset(), _golden_deal(), load_modality("biologic")
    tv = value_territory(asset, mod, load_territory(tid), DefaultLoAProvider(),
                         deal=deal, n_mc=3000, seed=777)
    exp_pie, exp_net, exp_lic, exp_p50, exp_ppos = GOLDEN[tid]

    # Base case is fully deterministic -> tight tolerance.
    assert tv.base.pie_rnpv == pytest.approx(exp_pie, rel=1e-6)
    assert tv.base.net_value == pytest.approx(exp_net, rel=1e-6)
    assert tv.base.deal.licensor_value == pytest.approx(exp_lic, rel=1e-6)
    # Deal split still reconciles to the pie.
    assert tv.base.deal.licensor_value + tv.base.deal.licensee_value == pytest.approx(tv.base.pie_rnpv)
    # Seeded Monte Carlo -> reproducible.
    assert tv.summaries["pie_rnpv"].p50 == pytest.approx(exp_p50, rel=1e-6)
    assert tv.summaries["pie_rnpv"].p_positive == pytest.approx(exp_ppos, abs=1e-6)
    assert tv.decision.recommendation == "GO"


def test_modality_switch_changes_revenue_shape():
    """Same indication as biologic (recurring) vs gene therapy (one_time) yields
    a front-loaded revenue shape for the one-time therapy."""
    territory = load_territory("brazil")
    epi = EpidemiologyBlock(basis="prevalence", rate_per_100k=500, diagnosis_rate=0.6,
                            treatment_rate=0.5, eligible_fraction=0.2)
    bio = AssetSpec(id="B", name="B", modality_id="biologic", indication="X", phase="phase3",
                    epidemiology=epi, global_list_price_usd=20000, exclusivity_years=10, peak_share=0.3)
    gt = AssetSpec(id="G", name="G", modality_id="gene_therapy", indication="X", phase="phase3",
                   epidemiology=epi, global_list_price_usd=1_500_000, exclusivity_years=12, peak_share=0.5)
    bio_tv = value_territory(bio, load_modality("biologic"), territory, DefaultLoAProvider(), n_mc=500, seed=1)
    gt_tv = value_territory(gt, load_modality("gene_therapy"), territory, DefaultLoAProvider(), n_mc=500, seed=1)

    def early_fraction(tv):
        t = tv.base.treated_patients
        return t[:5].sum() / t.sum()

    assert early_fraction(gt_tv) > early_fraction(bio_tv)
