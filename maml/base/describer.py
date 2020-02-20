import abc
import logging
from tqdm import tqdm
from typing import Any

import pandas as pd
from monty.json import MSONable
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
    a list of DataFrame/numpy.ndarray. This depends on the multioutput entry in
    the self._get_tags().
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

    def fit(self, objs, targets=None, **kwargs):
        """
        The fit function is used when describers have parameters that are dependent on the
        data.

        Args:
            objs (list): A list of objects.
            targets (list): Optional. A list of targets.
        """
        return self

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

        # process the outputs
        tags = self._get_tags()  # this is from BaseEstimator
        multi_output = tags['multioutput']

        if not multi_output:
            features = [features]

        is_pandas = hasattr(features[0][0], 'iloc')

        concated_features = [pd.concat(feature, keys=range(len(feature)), names=['input_index', None])
                          for feature in list(*zip(features))]

        if not is_pandas:
            concated_features = [features.values for features in concated_features]

        if multi_output:
            return concated_features
        else:
            return concated_features[0]


def _transform_one(describer: BaseDescriber, obj: Any):
    """
    A wrapper to make a pure function.
    """
    return describer.transform_one(obj)
