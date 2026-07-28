"""Sourced, typed input schemas.

Every number that enters a valuation is a :class:`SourcedValue`: a base value,
optional three-point bounds, a distribution kind, and **provenance** (where it
came from, how confident we are, as-of date). This is the mechanism that lets
`report.py` emit an assumptions ledger where each figure traces to a source or
an explicit assumption — the core defensibility requirement.

The models split cleanly by ownership of a fact:

* :class:`ModalityPack`  – facts true of a *modality* (dosing shape, COGS band).
* :class:`TerritoryPack` – facts true of a *country* (population, macro,
  regulatory timing, access defaults, pricing regime, cost-to-enter).
* :class:`AssetSpec`     – facts true of an *asset/indication* (epidemiology
  rates, price anchor, exclusivity, differentiation), plus optional
  ``territory_overrides`` for genuinely asset×territory inputs (e.g. the
  probability *this* drug gets reimbursed in *this* country, local competitors).
* :class:`DealTerms`     – the deal structure used to split the territory pie.

The resolver (:mod:`valuation_engine.inputs.resolve`) combines these into a flat
parameter set; nothing downstream imports pydantic.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from valuation_engine.core import distributions as D

Confidence = Literal["high", "medium", "low", "assumption"]
DistKind = Literal["point", "pert", "triangular", "bernoulli", "beta", "lognormal"]


class Provenance(BaseModel):
    """Where a value came from and how much to trust it."""

    model_config = ConfigDict(extra="forbid")

    source: Optional[str] = Field(
        default=None, description="URL or citation for the value"
    )
    quote: Optional[str] = Field(
        default=None, description="Verbatim supporting snippet from the source"
    )
    publisher: Optional[str] = None
    as_of: Optional[str] = Field(default=None, description="ISO date the value was current")
    method: Optional[str] = Field(
        default=None, description="How the value was derived (e.g. 'reference pricing', 'GBD 2021')"
    )
    confidence: Confidence = "assumption"
    notes: Optional[str] = None


class SourcedValue(BaseModel):
    """A single numeric input with distribution and provenance.

    ``kind`` selects the Monte Carlo distribution:

    * ``point``       – constant (``value``).
    * ``pert``        – beta-PERT over ``[low, value(mode), high]`` (default when
      ``low``/``high`` are supplied and ``kind`` is left as ``point``).
    * ``triangular``  – triangular over ``[low, value, high]``.
    * ``bernoulli``   – binary event with success probability ``value``.
    * ``beta``        – probability in [0,1] with mean ``value`` and
      concentration ``spread`` (default 20).
    * ``lognormal``   – positive skewed with median ``value`` and sigma ``spread``.
    """

    model_config = ConfigDict(extra="forbid")

    value: float
    low: Optional[float] = None
    high: Optional[float] = None
    kind: DistKind = "point"
    spread: Optional[float] = Field(
        default=None,
        description="Concentration (beta) or geometric sigma (lognormal)",
    )
    unit: Optional[str] = None
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _infer_and_check(self) -> "SourcedValue":
        # If bounds were given but kind left at the default, treat as PERT.
        if self.kind == "point" and (self.low is not None or self.high is not None):
            object.__setattr__(self, "kind", "pert")
        if self.kind in ("pert", "triangular"):
            lo = self.low if self.low is not None else self.value
            hi = self.high if self.high is not None else self.value
            if not (lo <= self.value <= hi):
                raise ValueError(
                    f"require low <= value <= high, got {lo} <= {self.value} <= {hi}"
                )
        if self.kind == "bernoulli" and not (0.0 <= self.value <= 1.0):
            raise ValueError("bernoulli value (probability) must be in [0, 1]")
        if self.kind == "beta" and not (0.0 <= self.value <= 1.0):
            raise ValueError("beta mean must be in [0, 1]")
        return self

    # -- conversions -------------------------------------------------------- #
    def base(self) -> float:
        return float(self.value)

    def distribution(self) -> D.Distribution:
        """Build the core Distribution for Monte Carlo sampling."""
        if self.kind == "point":
            return D.Deterministic(self.value)
        if self.kind == "pert":
            lo = self.low if self.low is not None else self.value
            hi = self.high if self.high is not None else self.value
            return D.PERT(lo, self.value, hi)
        if self.kind == "triangular":
            lo = self.low if self.low is not None else self.value
            hi = self.high if self.high is not None else self.value
            return D.Triangular(lo, self.value, hi)
        if self.kind == "bernoulli":
            return D.Bernoulli(self.value)
        if self.kind == "beta":
            return D.Beta(self.value, self.spread if self.spread else 20.0)
        if self.kind == "lognormal":
            return D.Lognormal(self.value, self.spread if self.spread else 0.3)
        raise ValueError(f"unknown kind {self.kind!r}")


def _coerce_sourced(v: object) -> object:
    """Allow a bare number in JSON to stand in for a point SourcedValue."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return {"value": float(v)}
    return v


# A field typed ``SV`` accepts either a number or a full SourcedValue mapping.
SV = Annotated[SourcedValue, BeforeValidator(_coerce_sourced)]

Phase = Literal["preclinical", "phase1", "phase2", "phase3", "filed", "approved"]
Dosing = Literal["recurring", "one_time"]
EpiBasis = Literal["incidence", "prevalence"]
DealType = Literal["own_commercialization", "out_license", "in_license", "distribution"]


class ModalityPack(BaseModel):
    """Facts true of a modality, independent of asset or geography."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    dosing: Dosing
    default_cogs_pct: SV
    # For recurring therapies: mean years a patient stays on treatment. For
    # one_time therapies this is 1 (a single administration).
    default_treatment_duration_years: SV = Field(
        default_factory=lambda: SourcedValue(value=1.0)
    )
    # For one_time (curative) therapies: annual replenishment of the treatable
    # prevalent pool as a fraction of the initial pool (new incident patients
    # after the prevalent bolus is drawn down). Ignored for recurring dosing.
    one_time_annual_replenishment: SV = Field(
        default_factory=lambda: SourcedValue(value=0.03)
    )
    cold_chain: bool = False
    notes: Optional[str] = None


class MacroBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gdp_per_capita_usd: SV
    health_exp_per_capita_usd: SV
    pct_gov_health: SV  # government share of health spend, 0..1


class TerritoryPack(BaseModel):
    """Facts true of a country, independent of the specific asset."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    iso3: str
    population: SV
    currency: str = "USD"
    fx_to_usd: SV = Field(default_factory=lambda: SourcedValue(value=1.0))
    macro: MacroBlock

    # Discounting: territory real discount rate = base_discount_rate + CRP.
    base_discount_rate: SV
    country_risk_premium: SV

    # Timing (years).
    regulatory_review_years: SV
    reimbursement_lag_years: SV

    # Access / "basket".
    p_reimbursement_default: SV  # probability of formulary/basket inclusion
    private_channel_share: SV  # fraction of treated pool reachable via self-pay
    public_gtn_discount: SV  # gross-to-net discount for public channel, 0..1
    private_gtn_discount: SV  # gross-to-net discount for private channel, 0..1

    # Affordability of the self-pay channel. The reference price at which self-pay
    # reach halves = affordability_multiple x GDP/capita; reach decays with a
    # logistic of steepness. Defaults apply if a pack omits them.
    affordability_multiple: SV = Field(default_factory=lambda: SourcedValue(value=2.0))
    affordability_steepness: SV = Field(default_factory=lambda: SourcedValue(value=2.5))

    # Pricing.
    reference_price_factor: SV  # net price as fraction of global anchor
    annual_price_erosion: SV  # annual % net-price decline pre-LoE

    # In-market commercialization economics (fraction of net sales). These
    # define the intrinsic operating "pie"; the deal-split layer allocates that
    # pie between licensor and licensee.
    sga_pct: SV = Field(default_factory=lambda: SourcedValue(value=0.25))
    distribution_pct: SV = Field(default_factory=lambda: SourcedValue(value=0.05))

    # Cost to enter/exploit the territory (USD).
    filing_cost_usd: SV
    market_access_spend_usd: SV
    bridging_trial_cost_usd: SV = Field(default_factory=lambda: SourcedValue(value=0.0))

    horizon_years: int = 20


class CompetitorEntry(BaseModel):
    """A competing asset expected in-territory."""

    model_config = ConfigDict(extra="forbid")

    name: str
    available_now: bool = False
    launch_year_from_now: SV = Field(default_factory=lambda: SourcedValue(value=0.0))
    # Mature share of the treated market this competitor captures (0..1).
    share_capture: SV = Field(default_factory=lambda: SourcedValue(value=0.0))


class TerritoryAssetInputs(BaseModel):
    """Asset×territory overrides — the outputs of drug-specific research.

    Any field set here overrides the corresponding territory default for this
    asset. All optional.
    """

    model_config = ConfigDict(extra="forbid")

    p_territory_approval: Optional[SV] = None
    p_reimbursement: Optional[SV] = None
    reimbursement_lag_years: Optional[SV] = None
    regulatory_review_years: Optional[SV] = None
    net_price_usd: Optional[SV] = None  # direct net price override (annual or per-course)
    reference_price_factor: Optional[SV] = None
    peak_share: Optional[SV] = None
    launch_delay_years: Optional[SV] = None  # extra delay (launch sequencing)
    market_access_spend_usd: Optional[SV] = None
    # Epidemiology funnel overrides — the single biggest driver of the pie, and
    # genuinely asset×territory (prevalence and treatment patterns differ by
    # country). Any set here overrides the asset's global EpidemiologyBlock for
    # this territory, so researched local data can flow straight into the model.
    epi_rate_per_100k: Optional[SV] = None
    diagnosis_rate: Optional[SV] = None
    treatment_rate: Optional[SV] = None
    eligible_fraction: Optional[SV] = None
    competitors: list[CompetitorEntry] = Field(default_factory=list)
    notes: Optional[str] = None


class EpidemiologyBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis: EpiBasis
    rate_per_100k: SV  # incidence or prevalence per 100,000 population
    diagnosis_rate: SV  # fraction diagnosed, 0..1
    treatment_rate: SV  # fraction of diagnosed who get drug therapy, 0..1
    eligible_fraction: SV = Field(
        default_factory=lambda: SourcedValue(value=1.0)
    )  # fraction of treated eligible for THIS asset


class AssetSpec(BaseModel):
    """The asset being valued, plus asset×territory overrides."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    modality_id: str
    indication: str
    phase: Phase
    epidemiology: EpidemiologyBlock

    # Global price anchor: annual net-equivalent for recurring therapies, or
    # per-course price for one_time therapies (USD, before territory factor).
    global_list_price_usd: SV

    # Exclusivity.
    exclusivity_years: SV  # years of on-market exclusivity from launch
    loe_erosion: SV = Field(
        default_factory=lambda: SourcedValue(value=0.2)
    )  # residual revenue fraction after loss of exclusivity

    # Competition / uptake defaults (can be overridden per territory).
    differentiation_score: SV = Field(
        default_factory=lambda: SourcedValue(value=1.0)
    )  # 1 = parity with SoC; >1 better (share & price premium); <1 worse
    peak_share: SV = Field(default_factory=lambda: SourcedValue(value=0.3))
    uptake_years_to_peak: SV = Field(default_factory=lambda: SourcedValue(value=5.0))

    # Clinical probability-of-success hook. If None, the ClinicalModelProvider
    # default LoA table is used. This is the seam for the user's future models.
    clinical_loa_override: Optional[SV] = None

    treatment_duration_years_override: Optional[SV] = None

    territory_overrides: dict[str, TerritoryAssetInputs] = Field(default_factory=dict)


class Milestone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    amount_usd: SV
    trigger: Literal["on_approval", "on_reimbursement", "at_year", "on_sales"]
    year_from_now: Optional[SV] = None  # required for at_year
    sales_threshold_usd: Optional[SV] = None  # required for on_sales
    # If set, the probability the milestone is reached (else derived from the
    # relevant gate: approval/reimbursement probability).
    probability_override: Optional[SV] = None


class RoyaltyTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_sales_usd: float = 0.0
    max_sales_usd: Optional[float] = None  # None = no upper bound
    rate: float  # royalty rate on net sales in this band, 0..1


class DealTerms(BaseModel):
    """Deal structure used to split the territory rNPV between parties."""

    model_config = ConfigDict(extra="forbid")

    deal_type: DealType = "out_license"
    upfront_usd: SV = Field(default_factory=lambda: SourcedValue(value=0.0))
    milestones: list[Milestone] = Field(default_factory=list)
    royalty_tiers: list[RoyaltyTier] = Field(default_factory=list)
