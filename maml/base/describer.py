import abc
import logging
from tqdm import tqdm
from typing import Any

import pandas as pd
from monty.json import MSONable
import numpy as np
from joblib import cpu_count, Parallel, delayed
from sklearn.utils.validation import check_memory
from sklearn.base import BaseEstimator, TransformerMixin


_ALLOWED_DATA = ('number', 'structure', 'molecule', 'spectrum')

_DESCRIBER_TYPES = ["composition", "site", "structure",
                    "general", "band_structure", "spectrum"]

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDescriber(BaseEstimator, TransformerMixin, MSONable, metaclass=abc.ABCMeta):
    """
    Base class for a Describer. A describer converts an object to a descriptor,
    typically a numerical representation useful for machine learning.
    The output for the describer can be a single DataFrame/numpy.ndarray or
    a list of DataFrame/numpy.ndarray.
    """

    def __init__(self, memory=None, verbose=True, n_jobs=0, **kwargs):
        """

        Args:
            memory (str/joblib.Memory): The path or Memory for caching the computational
                results, default None means no cache.
            verbose (bool): Whether to show the progress of feature calculations.
            n_jobs (int): The number of parallel jobs. 0 means no parallel computations.
                If this value is set to negative or greater than the total cpu
                then n_jobs is set to the number of cpu on system.
            **kwargs:
        """
        self.memory = check_memory(memory)
        self.verbose = verbose
        # find out the number of parallel jobs
        if (n_jobs < 0) or (n_jobs > cpu_count()):
            n_jobs = cpu_count()
            logger.info(f"Using {n_jobs} jobs for computation")
        self.n_jobs = n_jobs

    def transform_one(self, obj):
        """
        Transform an object.
        """
        raise NotImplementedError

    def transform(self, objs):
        """
        Transform a list of objs. If the return data is DataFrame,
        use df.xs(index, level='input_index') to get the result for the i-th object.

        Args:
            objs (list): A list of objects.
        Returns:
            One or a list of pandas data frame/numpy ndarray
        """
        cached_transform_one = self.memory.cache(_transform_one)

        if self.verbose:
            objs = tqdm(objs)

        # run the featurizer
        if self.n_jobs == 0:
            features = [cached_transform_one(self, obj) for obj in objs]
        else:
            features = Parallel(n_jobs=self.n_jobs)(
                delayed(cached_transform_one)(self, obj) for obj in objs)
        
        multi_output = self._is_multi_output()
        if not multi_output:
            features = [features]
        batched_features = [self._batch_features(i) for i in 
            list(*zip(features))]
        return batched_features if multi_output else batched_features[0]

    def _batch_features(self, features):
        """implement ways to combine list of features to one object.
        Default is simply return the original list
        
        Arguments:
            features (list): list of feature outputs from transform
        
        Returns:
            list of features
        """
        return features
    
    def _is_multi_output(self):
        tags = self._get_tags()
        multi_output = tags["multioutput"]  # this is from BaseEstimator
        return multi_output


class OutDataFrameConcat:
    def _batch_features(self, features):
        concated_features = pd.concat(features, 
            keys=range(len(features)), names=['input_index', None])
        return concated_features


class OutStackFirstDim:
    def _batch_features(self, features):
        return np.stack(features)


def _transform_one(describer: BaseDescriber, obj: Any):
    """
    A wrapper to make a pure function.
    """
    return describer.transform_one(obj)
