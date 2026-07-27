"""Addressable-patient math.

Bottom-up funnel: population -> epidemiology -> diagnosed -> treated -> eligible
for this asset. The *shape* of the treatable population over time depends on
dosing:

* recurring therapy on a **prevalence** basis -> a stock of prevalent patients
  each year;
* recurring therapy on an **incidence** basis -> steady-state on-therapy stock
  = annual incident patients x mean treatment duration;
* one_time (curative) therapy -> a prevalent **bolus** that is drawn down as
  patients are cured, plus incident replenishment. The drawdown is handled in
  the assembler (:mod:`valuation_engine.rnpv`) because it couples to uptake; the
  initial pool is computed here.
"""

from __future__ import annotations


def base_eligible(
    population: float,
    epi_rate_per_100k: float,
    diagnosis_rate: float,
    treatment_rate: float,
    eligible_fraction: float,
) -> float:
    """Patients eligible for this asset per the epidemiology funnel.

    Interpreted as a per-year quantity: a prevalent stock if the epi basis is
    prevalence, or an annual incident flow if the basis is incidence.
    """
    return (
        population
        * (epi_rate_per_100k / 1e5)
        * diagnosis_rate
        * treatment_rate
        * eligible_fraction
    )


def recurring_on_therapy_stock(
    base_eligible_value: float, basis: str, treatment_duration_years: float
) -> float:
    """Steady-state number of patients on therapy for a recurring drug.

    Prevalence basis: the eligible stock is already the on-therapy population.
    Incidence basis: multiply the annual incident flow by mean treatment
    duration (Little's law steady state).
    """
    if basis == "prevalence":
        return base_eligible_value
    if basis == "incidence":
        return base_eligible_value * max(treatment_duration_years, 0.0)
    raise ValueError(f"unknown epi basis {basis!r}")
