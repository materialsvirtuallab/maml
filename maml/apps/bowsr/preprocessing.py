"""
Module implements the scaler.
"""
from typing import List, Union

import numpy as np
from monty.json import MSONable


class StandardScaler(MSONable):
    """
    StandardScaler follows the sklean manner with addition of
    dictionary representation.
    """

    def __init__(self, mean: Union[List, np.ndarray] = None, std: Union[List, np.ndarray] = None):
        """
        Args:
            mean: np.ndarray, mean values
            std: np.ndnarray, standard deviations
        """
        self.mean = mean
        self.std = std

    def fit(self, target: Union[List, np.ndarray]) -> None:
        """
        Fit the StandardScaler to the target.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        mean = np.mean(target, axis=0)
        std = np.std(target, axis=0)
        self.mean = mean
        self.std = std

    def transform(self, target: np.ndarray) -> np.ndarray:
        """
        Transform target according to the mean and std.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        if self.mean is None or self.std is None:
            raise ValueError("No parameters is given.")
        return (target - self.mean) / self.std

    def inverse_transform(self, transformed_target: np.ndarray) -> np.ndarray:
        """
        Inversely transform the target.

        Args:
            transformed_target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        if self.mean is None or self.std is None:
            raise ValueError("No parameters is given.")
        return transformed_target * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

    def as_dict(self):
        """
        Dict representation of StandardScaler.
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "params": {"mean": self.mean.tolist(), "std": self.std.tolist()},
        }

        return d

    @classmethod
    def from_dict(cls, d):
        """
        Reconstitute a StandardScaler object from a dict representation of
        StandardScaler created using as_dict().

        Args
            d (dict): Dict representation of StandardScaler.
        """
        return cls(**d["params"])


class DummyScaler(MSONable):
    """
    Dummy scaler does nothing.
    """

    def fit(self, target: Union[List, np.ndarray]) -> None:
        """
        Fit the DummyScaler to the target.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        return

    def transform(self, target: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
        """
        Transform target.

        Args:
            target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        return target

    def inverse_transform(self, transformed_target: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
        """
        Inversely transform the target.

        Args:
            transformed_target (ndarray): The (mxn) ndarray. m is the number of samples,
                n is the number of feature dimensions.
        """
        return transformed_target

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def as_dict(self):
        """
        Serialize the instance into dictionary
        Returns:
        """
        d = {"@module": self.__class__.__module__, "@class": self.__class__.__name__, "params": {}}

        return d

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize from a dictionary
        Args:
            d: Dict, dictionary contain class initialization parameters

        Returns:

        """
        return cls()
