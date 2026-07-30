"""Load built-in territory and modality parameter packs from JSON.

Packs live in ``valuation_engine/territories/*.json`` and
``valuation_engine/modalities/*.json``. They are **seeded priors** — sourced
where possible, clearly confidence-tagged — intended as sensible defaults that
the research layer (Valyu + structured data sources) refines per asset. Nothing
here is a black box: every value is a ``SourcedValue`` with provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

from valuation_engine.inputs.schemas import AssetSpec, DealTerms, ModalityPack, TerritoryPack

_ROOT = Path(__file__).resolve().parent.parent
_TERRITORY_DIR = _ROOT / "territories"
_MODALITY_DIR = _ROOT / "modalities"
_ASSET_DIR = _ROOT / "assets"
_DEAL_DIR = _ROOT / "deals"


def load_territory(territory_id: str) -> TerritoryPack:
    path = _TERRITORY_DIR / f"{territory_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no territory pack for {territory_id!r} at {path}")
    return TerritoryPack.model_validate_json(path.read_text())


def load_modality(modality_id: str) -> ModalityPack:
    path = _MODALITY_DIR / f"{modality_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no modality pack for {modality_id!r} at {path}")
    return ModalityPack.model_validate_json(path.read_text())


def load_asset(asset_id: str) -> AssetSpec:
    """Load an asset from ``assets/<asset_id>.json`` — data, not code.

    An asset is just data like a territory or modality pack; valuing a new drug
    means adding a JSON file, never editing Python. Every input is a
    ``SourcedValue`` with provenance, so analyst assumptions are explicit and
    confidence-tagged rather than buried in a script.
    """
    path = _ASSET_DIR / f"{asset_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no asset pack for {asset_id!r} at {path}")
    return AssetSpec.model_validate_json(path.read_text())


def load_deal(deal_id: str) -> DealTerms:
    path = _DEAL_DIR / f"{deal_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no deal pack for {deal_id!r} at {path}")
    return DealTerms.model_validate_json(path.read_text())


def available_territories() -> list[str]:
    return sorted(p.stem for p in _TERRITORY_DIR.glob("*.json"))


def available_modalities() -> list[str]:
    return sorted(p.stem for p in _MODALITY_DIR.glob("*.json"))


def available_assets() -> list[str]:
    return sorted(p.stem for p in _ASSET_DIR.glob("*.json")) if _ASSET_DIR.exists() else []
