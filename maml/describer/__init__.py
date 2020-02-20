"""
Describer for converting (structural) objects into model-readable
numeric vectors or tensors.
"""

from .site import BispectrumCoefficients, SmoothOverlapAtomicPosition, \
    BPSymmetryFunctions
from .structure import DistinctSiteProperty, CoulombMatrix, \
    SortedCoulombMatrix, RandomizedCoulombMatrix
