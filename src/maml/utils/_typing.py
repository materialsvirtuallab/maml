"""Define several typing for convenient use."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
from pymatgen.core import Molecule, Structure

OptStrOrCallable = Optional[str | Callable[..., Any]]
StructureOrMolecule = Union[Structure, Molecule]
VectorLike = Union[list[float], np.ndarray]
