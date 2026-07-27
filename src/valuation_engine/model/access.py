"""Access / reimbursement ("basket") channel split.

Two channels reach the treated-eligible population:

* **Private / self-pay** — available from market entry, capped at
  ``private_channel_share`` of the pool (smaller, price-sensitive; the only
  channel before formulary listing). Meaningful in emerging markets.
* **Public / formulary** — the large channel, available only from
  ``public_start`` (approval + reimbursement lag) and only if the asset is
  actually reimbursed. Its reach is gated by ``p_reimbursement``: in the base
  case that is the probability (expected-value weight); in Monte Carlo it is a
  0/1 draw of whether the drug made the basket.

The two fractions are additive by construction — public covers the ``1 -
private_channel_share`` remainder — so blending the two net prices onto them
does not double-count patients.
"""

from __future__ import annotations

import numpy as np


def affordability_factor(
    net_price: np.ndarray | float,
    gdp_per_capita: float,
    affordability_multiple: float,
    steepness: float,
) -> np.ndarray | float:
    """Fraction of the self-pay segment that can actually afford the drug.

    The private/self-pay channel is real in these markets — but its reach
    collapses as the annual (or per-course) cost climbs past what the paying
    segment can sustain. We anchor an *affordability reference price* to local
    income: ``ref = affordability_multiple x gdp_per_capita`` (the private
    segment skews wealthy/insured, hence a multiple > 1). Reach then follows a
    logistic decay in price:

        factor = 1 / (1 + (net_price / ref) ** steepness)

    So a cheap drug keeps ~full self-pay reach (factor -> 1), a drug priced at
    the reference gets half, and a six-figure therapy is driven toward zero —
    the affordability wall. This is why an unreimbursed $300 pill and an
    unreimbursed $200k gene therapy are worth very different amounts out of
    pocket, even in the same "out-of-pocket" market.
    """
    ref = max(affordability_multiple * gdp_per_capita, 1e-9)
    ratio = np.asarray(net_price, dtype=float) / ref
    factor = 1.0 / (1.0 + np.power(ratio, steepness))
    return factor


def channel_fractions(
    years: np.ndarray,
    market_entry: float,
    public_start: float,
    private_channel_share: float,
    p_reimbursement: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(private_fraction, public_fraction)`` of the treated pool by year."""
    midyear = years + 0.5
    private = np.where(midyear >= market_entry, private_channel_share, 0.0)
    public = np.where(
        midyear >= public_start,
        (1.0 - private_channel_share) * p_reimbursement,
        0.0,
    )
    return private, public
