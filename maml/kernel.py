import numpy as np

from .utils.general import deserialize_maml_object


# now the kernels are defined as functions
# For future development the kernel should be class
# with tunable parameters, this is particular useful for Bayesian methods
def rbf(x1, x2, sigma):
    d_squared = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
    return np.exp(-d_squared / (2 * sigma ** 2))


def get_kernel(identifier):
    if isinstance(identifier, dict):
        config = {'class_name': identifier['class_name'], 'config': identifier['config']}
        return deserialize_maml_object(config, module_objects=globals())
    elif isinstance(identifier, str):
        return deserialize_maml_object(identifier, module_objects=globals())
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
