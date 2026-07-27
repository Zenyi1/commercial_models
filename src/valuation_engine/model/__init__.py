"""Domain model: pure functions that turn resolved parameters into the annual
drivers of a territory forecast (patients, price, share, channels, timing,
risk, costs).

Each function operates on plain floats and/or numpy year-arrays so it can be
unit-tested in isolation. The :mod:`valuation_engine.rnpv` assembler composes
them into the cash-flow forecast; the deal split lives in
:mod:`valuation_engine.model.deal`.
"""
