"""Core, domain-agnostic financial and statistical primitives.

Nothing in ``core`` knows about drugs or territories. It provides discounting
(``timeline``), probability distributions (``distributions``), and the Monte
Carlo / tornado machinery (``montecarlo``). Keeping these pure makes them
unit-testable against hand-computed known answers.
"""
