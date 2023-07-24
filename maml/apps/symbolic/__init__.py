"""Symbolic learning."""
from __future__ import annotations

try:
    import cvxpy as cp
except ImportError:
    cp = None

from ._feature_generator import FeatureGenerator, Operator
from ._selectors import SCAD, L0BrutalForce
from ._sis import ISIS, SIS

if cp is None:
    from ._selectors import AdaptiveLasso, DantzigSelector, Lasso
else:
    # import from cvxpy alternatives
    from ._selectors_cvxpy import AdaptiveLassoCP as AdaptiveLasso
    from ._selectors_cvxpy import DantzigSelectorCP as DantzigSelector
    from ._selectors_cvxpy import LassoCP as Lasso

__all__ = [
    "DantzigSelector",
    "AdaptiveLasso",
    "SCAD",
    "Lasso",
    "SIS",
    "ISIS",
    "L0BrutalForce",
    "FeatureGenerator",
    "Operator",
]
