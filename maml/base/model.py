import abc
from typing import Any, Union, List

import joblib
import numpy as np

from .describer import BaseDescriber


class BaseModel:
    """
    Abstract Base class for a Model. Basically, it usually wraps around a deep
    learning package, e.g., the Sequential Model in Keras, but provides for
    transparent conversion of arbitrary input and outputs.

    Args:
        describer (BaseDescriber): Describer that converts object into features
        model (Any): ML models, for example, sklearn model or keras model
    """
    def __init__(self, describer: BaseDescriber = None, model: Any = None, **kwargs):
        self.describer = describer
        self.model = model

    def fit(self, features: Union[List, np.ndarray],
            targets: Union[List, np.ndarray] = None, **kwargs):
        """
        Args:
            features (list or np.ndarray): Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
            targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
        """
        self.model.fit(features, targets, **kwargs)

    def _predict(self, features: np.ndarray):
        """
        Predict the values given a set of inputs based on fitted model.

        Args:
            features: array-like input features

        Returns:
            List of output objects
        """
        return self.model.predict(features)

    def predict_obj(self, obj: Any):
        return self._predict(self.describer.transform(obj))


class PotentialMixin:
    _model_type = 'potential'

    @abc.abstractmethod
    def predict_efs(self, include_force=False, include_stress=False):
        pass

    @abc.abstractmethod
    def run_md(self, time_step: float=1.0, n_steps: int=10000,
               temperature: float=300.0, ensemble: str='nve'):
        pass
