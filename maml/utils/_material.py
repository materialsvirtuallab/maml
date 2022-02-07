"""
Materials utils
"""
from typing import Union

from pymatgen.core import Composition, Molecule, Structure


def to_composition(obj: Union[Composition, Molecule, Structure, str]) -> Composition:
    """
    Convert str/structure or composition to compositions

    Args:
        obj (str/structure/composition): object to convert

    Returns:
        Composition object
    """
    if isinstance(obj, str):
        return Composition(obj)
    if isinstance(obj, (Structure, Molecule)):
        return obj.composition
    return obj
