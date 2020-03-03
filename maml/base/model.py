import joblib
from typing import Any, Union, List

import numpy as np
from sklearn.base import TransformerMixin


class BaseModel:
    """
    Abstract Base class for a Model. Basically, it usually wraps around a deep
    learning package, e.g., the Sequential Model in Keras, but provides for
    transparent conversion of arbitrary input and outputs.

    Args:
        describer (TransformerMixin): Describer that converts object into features
        model (Any): ML models, for example, sklearn model or keras model
    """
    def __init__(self, describer: TransformerMixin = None, model: Any = None, **kwargs):
        self.describer = describer
        self.model = model

    def fit(self, features: Union[List, np.ndarray],
            targets: Union[List, np.ndarray] = None, **kwargs) -> "BaseModel":
        """
        Args:
            features (list or np.ndarray): Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
            targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
        """
        self.model.fit(features, targets, **kwargs)
        return self

    def train(self,
              objs: Union[List, np.ndarray],
              targets: Union[List, np.ndarray], **kwargs) -> "BaseModel":
        features = self.describer.fit_transform(objs)
        return self.fit(features, targets, **kwargs)

    def _predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict the values given a set of inputs based on fitted model.

        Args:
            features (np.ndarray): array-like input features.

        Returns:
            List of output objects.
        """
        return self.model.predict(features, **kwargs)

    def predict_objs(self, objs: Union[List, np.ndarray]) -> np.ndarray:
        """
        Predict the values given a set of objects. Usually Pymatgen
            Structure objects.
        """
        return self._predict(self.describer.transform(objs))


class SklearnMixin:
    """
    Sklearn model save and load functionality
    """
    def save(self, filename: str):
        """Save the model and describer

        Arguments:
            filename (str): filename for save
        """
        joblib.dump({"model": self.model,
                     "describer": self.describer}, filename)

    def load(self, filename: str):
        """
        Load model parameters from filename
        Args:
            filename (str): model file name

        Returns: None

        """
        m = joblib.load(filename)
        self.model = m["model"]
        self.describer = m["describer"]

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        """
        Load the model from file
        Args:
            filename (str): filename
            **kwargs:

        Returns:

        """
        instance = cls(**kwargs)
        instance.load(filename)
        return instance


class KerasMixin:
    """keras model mixin with save and load functionality
    """
    def save(self, filename: str):
        joblib.dump(self.describer, filename)
        self.model.save(filename + '.hdf5')

    def load(self, filename: str):
        from keras.models import load_model
        self.describer = joblib.load(filename)
        self.model = load_model(filename + '.hdf5')

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        instance = cls(**kwargs)
        instance.load(filename)
        return instance


class ModelWithSklearn(BaseModel, SklearnMixin):
    """MAML model with sklearn model as estimator
    """
    pass


class ModelWithKeras(BaseModel, KerasMixin):
    """MAML model with keras model as estimators
    """
    pass
