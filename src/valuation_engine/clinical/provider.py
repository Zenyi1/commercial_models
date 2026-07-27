"""Clinical model plug-in interface.

The engine only needs one thing from a clinical model: a probability the asset
reaches market (likelihood-of-approval), as a ``SourcedValue``. Implement
:class:`ClinicalModelProvider` to wire in the user's models; the built-in
:class:`DefaultLoAProvider` returns benchmark values so the engine runs today.
"""

from __future__ import annotations

import abc

from valuation_engine.clinical.default_loa import default_loa
from valuation_engine.inputs.schemas import AssetSpec, Provenance, SourcedValue


class ClinicalModelProvider(abc.ABC):
    """Supplies the clinical likelihood-of-approval for an asset."""

    @abc.abstractmethod
    def probability_of_launch(self, asset: AssetSpec) -> SourcedValue:
        """Return P(asset reaches regulatory approval), as a Bernoulli-kind
        SourcedValue so Monte Carlo realises it as a 0/1 launch gate."""


class DefaultLoAProvider(ClinicalModelProvider):
    """Industry-benchmark LoA by phase and modality (placeholder)."""

    def probability_of_launch(self, asset: AssetSpec) -> SourcedValue:
        # An explicit override on the asset always wins.
        if asset.clinical_loa_override is not None:
            return asset.clinical_loa_override
        value, prov = default_loa(asset.phase, asset.modality_id)
        return SourcedValue(value=value, kind="bernoulli", provenance=Provenance(**prov))
