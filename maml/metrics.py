"""
Compute evaluation metrics
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .utils.general import deserialize_maml_object


def binary_accuracy(y_true, y_pred):
    """
    Compute the binary accuracy from true target and predicted target
    Args:
        y_true (list or np.ndarry): The true target labels
        y_pred (list or np.ndarray):  Predicted target labels

    Returns:
        float accuracy
    """
    return np.mean(np.array(y_true).ravel() == np.array(y_pred).ravel())


mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error


def get_metric(identifier):
    """
    Deserialize function metrics from identifier. The identifier can be object, string or dict

    Args:
        identifier (dict, str or callable): the identifier for the metric

    Returns:
        Callable metric function

    """
    if isinstance(identifier, dict):
        config = {'class_name': identifier['class_name'], 'config': identifier['config']}
        return deserialize_maml_object(config,
                                       module_objects=globals(),
                                       printable_module_name='metric function')
    elif isinstance(identifier, str):
        return deserialize_maml_object(identifier,
                                       module_objects=globals(),
                                       printable_module_name='metric function')
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
