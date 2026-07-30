"""Assets are data, not code: load from JSON and value via the CLI runner."""

from valuation_engine.inputs.packs import available_assets, load_asset, load_deal
from valuation_engine.inputs.schemas import AssetSpec, DealTerms
from valuation_engine import run


def test_dotinurad_asset_loads_from_json():
    a = load_asset("dotinurad")
    assert isinstance(a, AssetSpec)
    assert a.indication == "gout" and a.modality_id == "small_molecule"
    # The guessed inputs survive the round-trip AND stay explicitly tagged.
    assert a.epidemiology.eligible_fraction.value == 0.15
    assert a.epidemiology.eligible_fraction.provenance.confidence == "assumption"


def test_dotinurad_deal_loads_from_json():
    d = load_deal("dotinurad")
    assert isinstance(d, DealTerms)
    assert d.deal_type == "out_license" and d.upfront_usd.value == 2_000_000


def test_available_assets_lists_dotinurad():
    assert "dotinurad" in available_assets()


def test_cli_values_from_data_no_code_edit(capsys):
    # The whole point: a valuation with zero Python authored per run.
    run.main(["--asset", "dotinurad", "--territories", "brazil", "--n-mc", "200", "--seed", "1"])
    out = capsys.readouterr().out
    assert "Dotinurad" in out and "TERRITORY: Brazil" in out
    assert "research off" in out  # default path needs no research/API
