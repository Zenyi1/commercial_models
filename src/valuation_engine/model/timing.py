"""Launch timing and discounting — the levers that decide *when* cash flows
start, which under discounting is a first-order driver of value in emerging
markets (launch sequencing).
"""

from __future__ import annotations


def market_entry_year(regulatory_review_years: float, launch_delay_years: float) -> float:
    """Year from t=0 at which the product can first be sold (private channel).

    = regulatory review time + any launch-sequencing delay. Reimbursement lag
    is applied on top of this for the public channel (see ``public_start_year``).
    """
    return max(0.0, regulatory_review_years + launch_delay_years)


def public_start_year(market_entry: float, reimbursement_lag_years: float) -> float:
    """Year the public/formulary ("basket") channel can begin, conditional on
    the asset actually being reimbursed."""
    return market_entry + max(0.0, reimbursement_lag_years)


def discount_rate(base_discount_rate: float, country_risk_premium: float) -> float:
    """Territory real discount rate = base rate + country risk premium."""
    return base_discount_rate + country_risk_premium


def commercialization_weight(clinical_loa: float, p_territory_approval: float) -> float:
    """Overall probability the asset reaches the market in-territory.

    Product of the (injected) clinical likelihood-of-approval and the
    territory-level regulatory approval probability. In the base case both are
    probabilities (expected-value weight); in Monte Carlo both are 0/1 draws.
    """
    return clinical_loa * p_territory_approval
