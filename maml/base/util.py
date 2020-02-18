from typing import List
from sklearn.base import BaseEstimator
import pickle


def _check_objs_consistency(objs: List, type_names: List[str]):
    """
    Make sure the objs are consistent
    Args:
        objs: list of objects
        type_names: allowed type names in _ALLOWED_DATA
    Returns:
        whether the objects are consistent
    """
    types = set([i.__class__.__name__ for i in objs])
    if len(types) > 1:
        raise ValueError("Object types not consistent")

    type_test = types.pop()
    type_test = type_test.lower()
    if '.' in type_test:
        type_test = type_test.split('.')[-1]
    if type_test not in type_names:
        raise ValueError("Check types failed. %s not allowed in type_names" % type_test)

