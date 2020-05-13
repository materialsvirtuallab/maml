"""
Simple multi-layer perceptrons
"""
# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.


from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import tensorflow as tf

from maml import KerasModel, SequentialDescriber
from maml import BaseDescriber


def construct_mlp(hidden_layer_sizes: List[int],
                  input_dim: int,
                  lr: float = 1e-3,
                  activation: str = 'relu',
                  loss: str = 'mse',
                  metrics: List = None):
    """
    construct a minimal multi-layer perceptron model

    Args:
        hidden_layer_sizes (list of int): hidden layer neuron sizes
        input_dim (int): input dimension
        lr (float): learning rate
        activation (str): activation function
        loss (str): loss function
        metrics (list of str): metrics

    Returns:

    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=hidden_layer_sizes[0],
                                    input_dim=input_dim, activation=activation))
    for l in hidden_layer_sizes[1:]:
        model.add(tf.keras.layers.Dense(l, activation=activation))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr), metrics=metrics)
    return model


class MultiLayerPerceptron(KerasModel):
    """
    Multi-layer perceptron model with keras model as predictor
    """
    def __init__(self,
                 describer: BaseDescriber,
                 hidden_layer_sizes: List[int],
                 input_dim: int,
                 preprocessor: TransformerMixin = StandardScaler(),
                 activation: str = "relu",
                 loss: str = "mse",
                 **kwargs):
        """
        Args:
            describer (Describer): Describer to convert obj to features
            hidden_layer_sizes (list of int): hidden layer sizes for MLP
            input_dim (int): feature size
            preprocessor (TransformerMixin): sklearn transformer mixin that works
                on numerical values and outputs transformed values.
            activation (str): activation function for neural networks
            loss (str): loss function
        """

        describer = SequentialDescriber([describer, preprocessor])
        lr = kwargs.get('lr', 1e-3)
        metrics = kwargs.get("metrics", None)
        model = construct_mlp(hidden_layer_sizes, input_dim, lr, activation, loss, metrics)
        super().__init__(model=model, describer=describer)
