"""
Describer for converting (structural) objects into models-readable
numeric vectors or tensors.
"""
from __future__ import annotations

from ._composition import ElementProperty, ElementStats
from ._site import (
    BispectrumCoefficients,
    BPSymmetryFunctions,
    MEGNetSite,
    SiteElementProperty,
    SmoothOverlapAtomicPosition,
)
from ._structure import (
    CoulombEigenSpectrum,
    CoulombMatrix,
    DistinctSiteProperty,
    MEGNetStructure,
    RandomizedCoulombMatrix,
    SortedCoulombMatrix,
)

__all__ = [
    "BispectrumCoefficients",
    "SmoothOverlapAtomicPosition",
    "BPSymmetryFunctions",
    "DistinctSiteProperty",
    "CoulombMatrix",
    "SortedCoulombMatrix",
    "RandomizedCoulombMatrix",
    "ElementProperty",
    "ElementStats",
    "MEGNetSite",
    "MEGNetStructure",
    "CoulombEigenSpectrum",
    "SiteElementProperty",
]
