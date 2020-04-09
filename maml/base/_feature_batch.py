"""
Batch a list of features output by describer.transform method
"""
from typing import Any, List, Callable, Optional, Union

import pandas as pd
import numpy as np


def pandas_concat(features: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of pandas dataframe into a single one
    Args:
        features (list): list of pandas dataframe

    Returns: concatenated pandas dataframe

    """
    concatenated = pd.concat(features,
                             keys=range(len(features)),
                             names=['input_index', None])
    return concatenated


def stack_first_dim(features: List[np.ndarray]) -> np.ndarray:
    """
    Stack the first dimension. If the original features
    are a list of nxm array, the stacked features will be lxnxm,
    where l is the number of entries in the list
    Args:
        features (list): list of numpy array features

    Returns: stacked features

    """
    return np.stack(features)


def no_action(features: List[Any]) -> List[Any]:
    """
    return original feature lists

    """
    return features


def get_feature_batch(fb_name: Optional[Union[str, Callable]] = None) \
        -> Callable:  # type: ignore
    """
    Providing a feature batch name, returning the function callable
    Args:
        fb_name (str): name of the feature batch function
    Returns: callable feature batch function
    """

    if isinstance(fb_name, Callable):  # type: ignore
        return fb_name  # type: ignore

    if isinstance(fb_name, str):
        return globals()[fb_name]
    else:
        return no_action
