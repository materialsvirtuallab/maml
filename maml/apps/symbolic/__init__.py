"""
Symbolic learning
"""
try:
    import cvxpy as cp
except ImportError:
    cp = None

from ._feature_generator import FeatureGenerator, Operator  # noqa
from ._selectors import SCAD  # noqa
from ._selectors import L0BrutalForce  # noqa
from ._sis import ISIS, SIS  # noqa

if cp is None:
    from ._selectors import AdaptiveLasso, DantzigSelector, Lasso  # noqa  # noqa
else:
    # import from cvxpy alternatives
    from ._selectors_cvxpy import AdaptiveLassoCP as AdaptiveLasso  # noqa
    from ._selectors_cvxpy import DantzigSelectorCP as DantzigSelector  # noqa
    from ._selectors_cvxpy import LassoCP as Lasso  # noqa

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
