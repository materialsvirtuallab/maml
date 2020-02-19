import abc
import logging
from typing import Any

from joblib import cpu_count, Parallel, delayed
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_memory
from tqdm import tqdm

from .util import _check_objs_consistency


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDescriber(BaseEstimator, TransformerMixin,
                    metaclass=abc.ABCMeta):
    """
    Base class for a Describer, i.e., something that converts an object to a
    describer, typically a numerical representation useful for machine
    learning.
    The output for the describer can be a single DataFrame/np.ndarray or
    a list of DataFrame/np.ndarray. This depends on the multioutput entry in
    the self._get_tags()

    The BaseDescriber provides ways to

    """

    def __init__(self, memory=None, verbose=True, n_jobs=0, **kwargs):
        """

        Args:
            memory (None, str or joblib.Memory): provide path (str) or Memory for
                caching the computational results, default None means no cache
            verbose (bool): whether to show the progress of feature calculations
            n_jobs (int): number of parallel jobs. 0 means no parallel computations.
                If this value is set to negative or greater than the total cpu
                then n_jobs is set to the number of cpu on system
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
            objs: a list of objects
            targets: optional, a list of targets

        Returns: self
        """

        return self

    @abc.abstractmethod
    def transform_one(self, obj):
        pass

    def transform(self, objs):
        """
        Transform a list of objs. If the return data is DataFrame,
        use df.xs(index, level='input_index') to get the result
        for the i-th object

        Args:
            objs: a list of objects
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

        feature_temp = features[0][0]
        is_pandas = hasattr(feature_temp, 'iloc')

        features_final = [pd.concat(i, keys=range(len(i)), names=['input_index', None]
                                    ) for i in list(*zip(features))]

        if not is_pandas:
            features_final = [i.values for i in features_final]

        if multi_output:
            return features_final
        else:
            return features_final[0]


def _transform_one(describer: BaseDescriber, obj: Any):
    """
    Just a wrapper to make a pure function
    """
    return describer.transform_one(obj)
