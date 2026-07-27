"""Net-price trajectory over time.

Gross price is the global anchor scaled by the territory reference-price factor
(KSA external basket, Brazil CMED cap, Mexico procurement pressure are all
captured as a factor < 1). Net price applies the gross-to-net (access) discount,
then two erosion effects:

* **pre-LoE annual erosion** – ongoing reference-price/tender pressure;
* **loss of exclusivity** – at ``entry + exclusivity_years`` the franchise
  collapses to ``loe_erosion`` of its trajectory (generic / biosimilar entry).
"""

from __future__ import annotations

import numpy as np


def gross_price(list_price: float, reference_price_factor: float) -> float:
    return list_price * reference_price_factor


def net_price_trajectory(
    years: np.ndarray,
    market_entry: float,
    gross_price_value: float,
    gtn_discount: float,
    annual_erosion: float,
    exclusivity_years: float,
    loe_erosion: float,
) -> np.ndarray:
    """Per-period net price. Zero before market entry (mid-year timing)."""
    midyear = years + 0.5
    elapsed = np.maximum(0.0, midyear - market_entry)
    active = midyear >= market_entry

    base_net = gross_price_value * (1.0 - gtn_discount)
    price = base_net * np.power(max(1.0 - annual_erosion, 0.0), elapsed)

    loe_year = market_entry + exclusivity_years
    post_loe = midyear >= loe_year
    price = np.where(post_loe, price * loe_erosion, price)

    return np.where(active, price, 0.0)
