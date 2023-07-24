"""
Describer for converting (structural) objects into models-readable
numeric vectors or tensors.
"""
from __future__ import annotations

from ._composition import ElementProperty, ElementStats
from ._m3gnet import M3GNetStructure
from ._matminer import wrap_matminer_describer
from ._megnet import MEGNetSite, MEGNetStructure
from ._rdf import RadialDistributionFunction
from ._site import BispectrumCoefficients, BPSymmetryFunctions, SiteElementProperty, SmoothOverlapAtomicPosition
from ._structure import (
    CoulombEigenSpectrum,
    CoulombMatrix,
    DistinctSiteProperty,
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
    "wrap_matminer_describer",
    "M3GNetStructure",
    "MEGNetSite",
    "MEGNetStructure",
    "RadialDistributionFunction",
]
