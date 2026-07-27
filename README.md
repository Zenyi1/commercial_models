# Territorial Rights Valuation Engine

A Python engine that puts a **defensible, risk-adjusted number** on the
commercial value of the *territorial rights* to a biotech asset — any modality
(small molecule, biologic, gene therapy), any stage (preclinical → Phase 1–3 →
approved), per geography. Ships with sourced packs for **Mexico, Brazil, and
Saudi Arabia**.

The output is what goes in front of a CEO / PE head: a base-case rNPV, a Monte
Carlo P10/P50/P90 range with the probability the rights clear zero, a tornado of
value drivers, a deal-structure split, a go/no-go with break-evens, an
independent triangulation, and an **assumptions ledger where every number traces
to a source or an explicit assumption**.

## Principles

- **Data & math, never LLM guesses.** LLMs/tools only find and structure
  *sourced* evidence into `SourcedValue`s; all arithmetic lives in auditable code.
- **One code path.** The base case and Monte Carlo run the *same* assembler —
  probabilities are expected-value weights in the base case and 0/1 draws under
  simulation, so the two can't drift apart.
- **Pluggable.** Clinical trial-success risk is out of scope and injected via
  `ClinicalModelProvider`; evidence comes through a `ResearchProvider` interface
  (Valyu is one backend among crawlers and structured data feeds).

## What it models (per territory)

Time-to-market (regulatory review + reimbursement lag) → addressable patients
(with gene-therapy prevalent-pool drawdown) → access channels (private/self-pay
vs public "basket", gated by reimbursement probability) → net price (reference
pricing, erosion, loss of exclusivity) → competition-adjusted uptake (in-market
differentiation + competitor pipeline entry) → risk-weighting (injected clinical
LoA × territory approval) → discounting (WACC + country risk premium) → **rNPV**,
then a deal-split layer (upfront + milestones + tiered royalty → licensor vs
licensee value) and a cost-to-enter go/no-go.

## Quick start

```bash
python -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
PYTHONPATH=src python examples/value_asset.py    # full report across MX/BR/SA
./.venv/bin/python -m pytest -q                  # 60 tests
```

Minimal use:

```python
from valuation_engine.clinical.provider import DefaultLoAProvider
from valuation_engine.inputs.packs import load_modality, load_territory
from valuation_engine.inputs.schemas import AssetSpec, EpidemiologyBlock, SourcedValue
from valuation_engine.report import value_territory, render_text

asset = AssetSpec(
    id="X", name="X", modality_id="biologic", indication="RA", phase="phase3",
    epidemiology=EpidemiologyBlock(basis="prevalence", rate_per_100k=500,
        diagnosis_rate=0.6, treatment_rate=0.5, eligible_fraction=0.2),
    global_list_price_usd=SourcedValue(value=20000, low=15000, high=25000),
    exclusivity_years=10, peak_share=0.25,
)
tv = value_territory(asset, load_modality("biologic"), load_territory("brazil"),
                     DefaultLoAProvider())
print(render_text(tv))
```

## Layout

```
src/valuation_engine/
  core/         discounting, distributions, Monte Carlo + tornado
  model/        epidemiology, pricing, access, competition, uptake, costs, timing, deal
  inputs/       SourcedValue schemas, resolver, pack loader
  territories/  mexico / brazil / saudi_arabia .json  (sourced priors)
  modalities/   small_molecule / biologic / gene_therapy .json
  research/     ResearchProvider interface, cached/valyu/web + data_sources adapters
  clinical/     ClinicalModelProvider interface + default LoA table (placeholder)
  comparables/  deal schema, harvest, analyzer, triangulation
  rnpv.py       the assembler   ·   report.py   the orchestrator + rendering
data/comparables/deals.json     seed corpus (illustrative — replace via harvest)
examples/value_asset.py         end-to-end reference driver
```

## Status of the numbers

Territory/modality packs are **sourced priors**, confidence-tagged, meant to be
refined per asset by the research layer. The comparables corpus shipped here is
**illustrative and labelled as such** — wire `SecEdgarDealsSource` /
`CommercialDbDealsSource` and run `comparables.harvest` to replace it with
primary-source data.
