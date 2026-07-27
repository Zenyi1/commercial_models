"""Structured, sourced data adapters.

Each adapter is a :class:`~valuation_engine.research.provider.ResearchProvider`
that answers a specific set of canonical keys from an authoritative feed, so a
number traces to data rather than judgment:

* :class:`EpiMacroSource`   – WHO/IHME epidemiology, World Bank GDP + health
  expenditure/affordability, Damodaran CRP, FX  → patients, discount rate.
* :class:`RegulatoryIpSource` – Drugs@FDA/openFDA, EMA, COFEPRIS/ANVISA/SFDA
  registries, patent/Orange-Purple Book → review time, approval status, LoE.
* :class:`HtaSource`        – ICER/NICE/CADTH/IQWiG + CONITEC/CSG/SFDA-EES
  → reimbursement probability + value-based price anchor.
* :class:`PricingSource`    – national price lists + intl reference pricing
  → net price / reference-price factor.
* :class:`TrialsLitSource`  – ClinicalTrials.gov + PubMed/Cochrane
  → competitor pipeline & differentiation.
* :class:`DealsSource`      – SEC EDGAR + commercial deal DBs → the comparables
  corpus (used by ``comparables.harvest``).

All are config-gated stubs: ``available()`` is False until a backend is wired,
and ``get`` returns None so the aggregator falls back cleanly. Each ``_fetch``
is the localized wiring point and must return a ``SourcedValue`` with provenance.
"""

from __future__ import annotations

import abc
import os
from typing import Optional

from valuation_engine.inputs.schemas import SourcedValue
from valuation_engine.research.provider import ResearchProvider, ResearchQuery


class StructuredAdapter(ResearchProvider):
    """Base for structured feeds: answers a fixed set of keys when configured."""

    handled_keys: frozenset[str] = frozenset()
    env_flag: str = ""  # env var that enables the adapter

    def available(self) -> bool:
        return bool(self.env_flag and os.getenv(self.env_flag))

    def get(self, query: ResearchQuery) -> Optional[SourcedValue]:
        if query.key not in self.handled_keys or not self.available():
            return None
        return self._fetch(query)

    @abc.abstractmethod
    def _fetch(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        ...


class EpiMacroSource(StructuredAdapter):
    name = "epi_macro"
    env_flag = "EPI_MACRO_ENABLED"
    handled_keys = frozenset(
        {"epi_rate_per_100k", "diagnosis_rate", "treatment_rate", "eligible_fraction"}
    )

    def _fetch(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        raise NotImplementedError("Wire WHO/IHME + World Bank feeds here.")


class RegulatoryIpSource(StructuredAdapter):
    name = "regulatory_ip"
    env_flag = "REGULATORY_IP_ENABLED"
    handled_keys = frozenset(
        {"regulatory_review_years", "p_territory_approval", "launch_delay_years"}
    )

    def _fetch(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        raise NotImplementedError("Wire Drugs@FDA/EMA/registries + patent data here.")


class HtaSource(StructuredAdapter):
    name = "hta"
    env_flag = "HTA_ENABLED"
    handled_keys = frozenset({"p_reimbursement", "reimbursement_lag_years", "net_price_usd"})

    def _fetch(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        raise NotImplementedError("Wire ICER/NICE/CADTH + CONITEC/CSG/SFDA-EES here.")


class PricingSource(StructuredAdapter):
    name = "pricing"
    env_flag = "PRICING_ENABLED"
    handled_keys = frozenset({"reference_price_factor", "net_price_usd"})

    def _fetch(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        raise NotImplementedError("Wire national price lists + intl reference pricing here.")


class TrialsLitSource(StructuredAdapter):
    name = "trials_lit"
    env_flag = "TRIALS_LIT_ENABLED"
    handled_keys = frozenset({"peak_share"})  # differentiation-informed share ceiling

    def _fetch(self, query: ResearchQuery) -> Optional[SourcedValue]:  # pragma: no cover
        raise NotImplementedError("Wire ClinicalTrials.gov + PubMed/Cochrane here.")


__all__ = [
    "StructuredAdapter",
    "EpiMacroSource",
    "RegulatoryIpSource",
    "HtaSource",
    "PricingSource",
    "TrialsLitSource",
]
