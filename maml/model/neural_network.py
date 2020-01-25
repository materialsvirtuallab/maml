# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from maml.abstract import Model


class MultiLayerPerceptron(Model):
    """
    Basic neural network model.
    """

    def __init__(self, layer_sizes, describer, preprocessor=None,
                 activation="relu", loss="mse"):
        """
        Args:
            layer_sizes (list): Hidden layer sizes, e.g., [3, 3].
            describer (Describer): Describer to convert
                input objects to descriptors.
            preprocessor (BaseEstimator): Processor to use.
                Defaults to StandardScaler
            activation (str): Activation function
            loss (str): Loss function. Defaults to mae
        """
        self.layer_sizes = layer_sizes
        self.describer = describer
        self.output_describer = None
        self.preprocessor = preprocessor
        self.activation = activation
        self.loss = loss
        self.model = None

    def fit(self, inputs, outputs, test_size=0.2, adam_lr=1e-2, **kwargs):
        """
        Args:
            inputs (list): List of inputs
            outputs (list): List of outputs
            test_size (float): Size of test set. Defaults to 0.2.
            adam_lr (float): learning rate of Adam optimizer
            kwargs: Passthrough to fit function in keras.models
        """
        from keras.optimizers import Adam
        from keras.models import Sequential
        from keras.layers import Dense
        descriptors = self.describer.transform(inputs)
        if self.preprocessor is None:
            self.preprocessor = StandardScaler()
            scaled_descriptors = self.preprocessor.fit_transform(descriptors)
        else:
            scaled_descriptors = self.preprocessor.transform(descriptors)
        adam = Adam(adam_lr)
        x_train, x_test, y_train, y_test = train_test_split(
            scaled_descriptors, outputs, test_size=test_size)

        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_dim=len(x_train[0]),
                        activation=self.activation))
        for l in self.layer_sizes[1:]:
            model.add(Dense(l, activation=self.activation))
        model.add(Dense(1))
        model.compile(loss=self.loss, optimizer=adam, metrics=[self.loss])
        model.fit(x_train, y_train, verbose=0, validation_data=(x_test, y_test),
                  **kwargs)
        self.model = model

    def predict(self, inputs):
        """
        Predict outputs with fitted model.

        Args:
            inputs (list): List of input testing objects.
        """
        descriptors = self.describer.transform(inputs)
        scaled_descriptors = self.preprocessor.transform(descriptors)
        outputs = self.model.predict(scaled_descriptors)
        return outputs

    def save(self, model_fname, scaler_fname):
        """
        Use kears model.save method to save model in *.h5 file
        Use scklearn.external.joblib to save scaler(the *.save
        file is supposed to be much smaller than saved as
        pickle file)

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
