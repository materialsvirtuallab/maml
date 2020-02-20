"""
Linear models
"""
# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import joblib
from maml import BaseModel
from sklearn import linear_model


class LinearModel(BaseModel):
    """
    Linear model.
    """
    def __init__(self, describer, regressor="LinearRegression", **kwargs):
        """
        Args:
            describer (BaseDescriber): Describer to convert objects to descriptors.
            regressor (str): Name of LinearModel from sklearn.linear_model.
                Default to "LinearRegression", i.e., ordinary least squares.
            kwargs: kwargs to be passed to regressor.
        """
        self.describer = describer
        self.regressor = regressor
        self.kwargs = kwargs
        lr = getattr(linear_model, regressor)
        self.model = lr(**kwargs)
        super().__init__(describer=describer)

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
