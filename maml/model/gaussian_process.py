# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import warnings

import joblib
from maml.abstract import Model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels


class GaussianProcessRegressionModel(Model):
    """
    Gaussian Process Regression Model

    """

    def __init__(self, describer, kernel_category='RBF', restarts=10, **kwargs):
        """

        Args:
            describer (Describer): Describer to convert
                input object to descriptors.
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
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts)
        self._xtrain = None
        self._xtest = None

    def fit(self, inputs, outputs, override=False):
        """
        Args:
            inputs (list): List of input training objects.
            outputs (list): List/Array of output values
                (supervisory signals).
            override: (bool) Whether to calculate the feature
                vectors from given inputs. Default to False. Set to True if
                you want to retrain the model with a different set of
                training inputs.
        """
        if not self._xtrain or override:
            xtrain = self.describer.describe_all(inputs)
        else:
            warnings.warn("Feature vectors retrieved from cache "
                          "and input training objects ignored. "
                          "To override the old cache with feature vectors "
                          "of new training objects, set override=True.")
            xtrain = self._xtrain
        self.model.fit(xtrain, outputs)
        self._xtrain = xtrain

    def predict(self, inputs, override=False, **kwargs):
        """
        Args:
            inputs (List): List of input testing objects.
            override: (bool) Whether to calculate the feature
                vectors from given inputs. Default to False. Set to True if
                you want to test the model with a different set of testing inputs.
            kwargs: kwargs to be passed to predict method, e.g.
                return_std, return_cov.
        Returns:
            Predicted output array from inputs.
        """
        if self._xtest is None or override:
            xtest = self.describer.describe_all(inputs)
        else:
            warnings.warn("Feature vectors retrieved from cache "
                          "and input testing objects ignored. "
                          "To override the old cache with feature vectors "
                          "of new testing objects, set override=True.")
            xtest = self._xtest
        self._xtest = xtest
        return self.model.predict(xtest, **kwargs)

    @property
    def params(self):
        return self.model.get_params()

    def save(self, model_fname):
        joblib.dump(self.model, '%s.pkl' % model_fname)

    def load(self, model_fname):
        self.model = joblib.load(model_fname)
