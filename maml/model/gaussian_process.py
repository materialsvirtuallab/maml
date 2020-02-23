"""
Gaussian process models
"""
# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import joblib
from maml import BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor, kernels


class GaussianProcessRegressionModel(BaseModel):
    """
    Gaussian Process Regression Model.
    """
    def __init__(self, describer, kernel_category='RBF', restarts=10, **kwargs):
        """
        Args:
            describer (BaseDescriber): Describer to convert input object to
                descriptors.
            kernel_category (str): Name of kernel from
                sklearn.gaussian_process.kernels. Default to 'RBF', i.e.,
                squared exponential.
            restarts (int): The number of restarts of the optimizer for
                finding the kernel’s parameters which maximize the
                log-marginal likelihood.
            kwargs: kwargs to be passed to kernel object, e.g. length_scale,
                length_scale_bounds.
        """
        self.describer = describer
        kernel = getattr(kernels, kernel_category)(**kwargs)
        model = GaussianProcessRegressor(kernel=kernel,
                                         n_restarts_optimizer=restarts)
        self.model = model
        super().__init__(describer=describer, model=model)

    @property
    def params(self):
        return self.model.get_params()

    def save(self, model_fname):
        joblib.dump(self.model, '%s.pkl' % model_fname)

    def load(self, model_fname):
        self.model = joblib.load(model_fname)
