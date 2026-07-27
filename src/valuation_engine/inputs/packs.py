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

from valuation_engine.inputs.schemas import ModalityPack, TerritoryPack

_ROOT = Path(__file__).resolve().parent.parent
_TERRITORY_DIR = _ROOT / "territories"
_MODALITY_DIR = _ROOT / "modalities"


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


def available_territories() -> list[str]:
    return sorted(p.stem for p in _TERRITORY_DIR.glob("*.json"))


def available_modalities() -> list[str]:
    return sorted(p.stem for p in _MODALITY_DIR.glob("*.json"))
