"""
Wrapper for matminer featurizers
"""

from inspect import signature
import logging
from typing import Any

import pandas as pd

from maml.base import BaseDescriber, OutDataFrameConcat

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def wrap_matminer_describer(cls_name: str, wrapped_class: Any):
    """
    Wrapper of matminer describers.
    Args:
        cls_name (str): new class name
        wrapped_class (class object): matminer BaseFeaturizer
    Returns: maml describer class
    """
    new_class = type(cls_name, (OutDataFrameConcat, BaseDescriber),
                     {'__doc__': wrapped_class.__doc__})

    def __init__(self, *args, **kwargs):
        """
        Wrapped __init__ constructor
        """
        n_jobs = kwargs.pop("n_jobs", 0)
        memory = kwargs.pop("memory", None)
        verbose = kwargs.pop("verbose", False)
        wrapped_class.__init__(self, *args, **kwargs)
        logger.info(f"Using matminer {wrapped_class.__name__} class")
        base_kwargs = dict(n_jobs=n_jobs, memory=memory, verbose=verbose)
        super(new_class, self).__init__(**base_kwargs)

    new_class.__init__ = __init__  # type: ignore

    def transform_one(self, obj: Any):
        """
        featurize to transform_one
        """
        results = wrapped_class.featurize(self, obj)
        labels = wrapped_class.feature_labels(self)
        return pd.DataFrame({i: [j] for i, j in zip(labels, results)})

    new_class.transform_one = transform_one  # type: ignore

    def from_preset(name: str, **kwargs):
        """
        Wrap matminer's from_preset function
        """
        instance = wrapped_class.from_preset(name)
        sig = signature(wrapped_class.__init__)
        args = list(sig.parameters.keys())[1:]
        params = {i: None for i in args}
        params.update(**kwargs)
        instance_new = new_class(**params)
        instance_new.__dict__.update(instance.__dict__)
        return instance_new
    new_class.from_preset = from_preset  # type: ignore

    return new_class
