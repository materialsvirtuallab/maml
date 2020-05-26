"""
Materials utils
"""
from typing import Union
from pymatgen.core import Composition, Structure


def to_composition(obj: Union[Composition, Structure, str]) -> Composition:
    """
    Convert str/structure or composition to compositions

    Args:
        obj (str/structure/composition): object to convert

    Returns:
        Composition object
    """
    if isinstance(obj, str):
        return Composition(obj)
    elif isinstance(obj, Structure):
        return obj.composition
    return obj
