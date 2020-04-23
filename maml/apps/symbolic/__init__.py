"""
Symbolic learning
"""
try:
    import cvxpy as cp
except ImportError:
    cp = None

from ._selectors import SCAD  # noqa
from ._feature_generator import features_generator, Operator  # noga

if cp is None:
    from ._selectors import (DantzigSelector, AdaptiveLasso,  # noqa
                             Lasso)  # noqa
else:
    # import from cvxpy alternatives
    from ._selectors_cvxpy import DantzigSelectorCP as DantzigSelector  # noqa
    from ._selectors_cvxpy import AdaptiveLassoCP as AdaptiveLasso  # noqa
    from ._selectors_cvxpy import LassoCP as Lasso  # noqa

from ._sis import SIS  # noqa

__all__ = [
    "DantzigSelector",
    "AdaptiveLasso",
    "SCAD",
    "Lasso",
    "SIS",
    "features_generator",
    "Operator"
]
