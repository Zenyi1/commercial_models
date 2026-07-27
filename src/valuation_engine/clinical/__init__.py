"""Clinical probability-of-success plug-in.

Clinical / technical trial-success risk is **out of scope** for this engine —
the user maintains those models separately. This package defines the seam:
:class:`ClinicalModelProvider.probability_of_launch` returns a
``SourcedValue`` likelihood-of-approval that the rNPV engine consumes as a risk
weight. The built-in :class:`DefaultLoAProvider` supplies industry benchmark
LoA rates as a placeholder; swap in the user's model without touching the core.
"""
