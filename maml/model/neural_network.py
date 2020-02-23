"""
Simple multi-layer perceptrons
"""
# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import joblib
import numpy as np
from maml import BaseModel
from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MultiLayerPerceptron(BaseModel):
    """
    Neural network model.
    """

    def __init__(self, describer, layer_sizes, preprocessor=StandardScaler(),
                 activation="relu", loss="mse"):
        """
        Args:
            describer (Describer): Describer to convert input objects to
                descriptors.
            layer_sizes (list): Hidden layer sizes, e.g., [3, 3].
            activation (str): Activation function
            loss (str): Loss function. Defaults to mae
        """
        self.describer = describer
        self.layer_sizes = layer_sizes
        self.preprocessor = preprocessor
        self.activation = activation
        self.loss = loss
        self.model = None
        super().__init__(describer=describer)

    def fit(self, features, targets, test_size=0.2, adam_lr=1e-2, **kwargs):
        """
        Args:
            features (list or np.ndarray): Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
            targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
            test_size (float): Size of test set. Defaults to 0.2.
            adam_lr (float): learning rate of Adam optimizer
            kwargs: Passthrough to fit function in keras.models
        """
        from keras.layers import Dense
        from keras.optimizers import Adam
        from keras.models import Sequential
        scaled_features = self.preprocessor.fit_transform(features)
        x_train, x_test, y_train, y_test \
            = train_test_split(scaled_features, targets, test_size=test_size)

        model = Sequential()
        model.add(Dense(units=self.layer_sizes[0], input_dim=len(x_train[0]),
                        activation=self.activation))
        for l in self.layer_sizes[1:]:
            model.add(Dense(l, activation=self.activation))
        model.add(Dense(1))
        model.compile(loss=self.loss, optimizer=Adam(adam_lr), metrics=[self.loss])
        model.fit(x_train, y_train, verbose=0, validation_data=(x_test, y_test),
                  **kwargs)
        self.model = model

    def _predict(self, features: np.ndarray, **kwargs):
        """
        Args:
            features (np.ndarray): array-like input features.
        """
        scaled_features = self.preprocessor.transform(features)
        return self.model.predict(scaled_features, **kwargs)

    def predict_obj(self, objs: Any):
        """
        Predict the values given a set of objects. Usually Pymatgen
            Structure objects.
        """
        return self._predict(self.preprocessor.transform(self.describer.transform(objs)))

    def save(self, model_fname, scaler_fname):
        """
        Use kears model.save method to save model in *.h5 file
        Use sklearn.external.joblib to save scaler (the *.save
        file is supposed to be much smaller than saved as pickle file)

        Args:
            model_fname (str): Filename to save model object.
            scaler_fname (str): Filename to save scaler object.
        """
        self.model.save(model_fname)
        joblib.dump(self.preprocessor, scaler_fname)

    def load(self, model_fname, scaler_fname):
        """
        Load model and scaler from corresponding files.

        Args:
            model_fname (str): Filename storing model.
            scaler_fname (str): Filename storing scaler.
        """
        from keras.models import load_model
        self.model = load_model(model_fname)
        self.preprocessor = joblib.load(scaler_fname)
