"""Typed, sourced input schemas and the resolver that flattens them.

``schemas`` defines the provenance-carrying pydantic models that describe an
asset, a modality pack, a territory pack and a deal. ``resolve`` merges those
into the flat, plain-number :class:`~valuation_engine.inputs.resolve.ResolvedInputs`
that the deterministic model consumes.
"""
