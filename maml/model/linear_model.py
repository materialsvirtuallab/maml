# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import warnings

import joblib
from sklearn import linear_model
from maml.abstract import Model


class LinearModel(Model):
    """
    Linear model.
    """

    def __init__(self, describer, regressor="LinearRegression", **kwargs):
        """

        Args:
            describer (Describer): Describer to convert structure objects
                to descriptors.
            regressor (str): Name of LinearModel from sklearn.linear_model.
                Default to "LinearRegression", i.e., ordinary least squares.
            kwargs: kwargs to be passed to regressor.
        """
        self.describer = describer
        self.regressor = regressor
        self.kwargs = kwargs
        lr = getattr(linear_model, regressor)
        self.model = lr(**kwargs)
        self._xtrain = None
        self._xtest = None

    def fit(self, inputs, outputs, weights=None, override=False):
        """
        Fit model.

        Args:
            inputs (list): List of input training objects.
            outputs (list): List/Array of output values
                (supervisory signals).
            weights (list): List/Array of weights. Default to None,
                i.e., unweighted.
            override (bool): Whether to calculate the feature vectors
                from given inputs. Default to False. Set to True if
                you want to retrain the model with a different set of
                training inputs.
        """
        if self._xtrain is None or override:
            xtrain = self.describer.describe_all(inputs)
        else:
            warnings.warn("Feature vectors retrieved from cache "
                          "and input training objects ignored. "
                          "To override the old cache with feature vectors "
                          "of new training objects, set override=True.")
            xtrain = self._xtrain
        self.model.fit(xtrain, outputs, weights)
        self._xtrain = xtrain

    def predict(self, inputs, override=False):
        """
        Predict outputs with fitted model.

        Args:
            inputs (list): List of input testing objects.
            override (bool): Whether to calculate the feature
                vectors from given inputs. Default to False. Set to True
                if you want to test the model with a different set of
                testing inputs.

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
        return self.model.predict(xtest)

    def evaluate_fit(self):
        """
        Efficient method to obtain prediction on training inputs w/o
        calculating the features of inputs again.

        Args:
            Predicted output array from training inputs.
        """
        self._xtest = self._xtrain
        return self.predict(inputs=None, override=False)

    @property
    def coef(self):
        return self.model.coef_

    @property
    def intercept(self):
        return self.model.intercept_

    def save(self, model_fname):
        joblib.dump(self.model, '%s.pkl' % model_fname)

    def load(self, model_fname):
        self.model = joblib.load(model_fname)
