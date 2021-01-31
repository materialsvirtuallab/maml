"""Deep learning layers"""
from typing import Optional, Sequence

import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.layers import Layer
from megnet.layers.readout import Set2Set
from megnet.utils.layer import repeat_with_index


class WeightedAverageLayer(Layer):
    r"""
    Weight average the features using weights

    result= \sum{w_i^a * value_i} / \sum{w_i^a}

    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        """
        Args:
            alpha (float): exponent in weighting
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape: Sequence) -> None:
        """
        Build the layer

        Args:
            input_shape (tuple): input shape tuple
        """
        self.built = True

    def call(self, inputs: Sequence, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Core logic of the layer

        Args:
            inputs (tuple): input tuple of length 3
            mask (tf.Tensor): not used here
        """
        # prop: [1, n, n_feature]
        # weight: [1, n]
        # index： [1, n]
        prop, weights, index = inputs
        expo_weights = weights ** self.alpha
        prop = prop * expo_weights[:, :, None]
        prop_sum = self.reduce_sum(prop, index, perm=[1, 0, 2])
        weight_sum = self.reduce_sum(expo_weights, index, perm=[1, 0])
        return prop_sum / weight_sum[:, :, None]

    @staticmethod
    def reduce_sum(prop: tf.Tensor, index: tf.Tensor, perm: Sequence) -> tf.Tensor:
        """
        Reduce sum the tensors using index

        Args:
            prop (tf.Tensor): tensor with shape [1, n, ...]
            index (tf.Tensor): integer tensor with shape [1, n]
            perm (list): permutation for transpose.
        """
        index = tf.reshape(index, (-1,))
        prop = tf.transpose(a=prop, perm=perm)
        out = tf.math.segment_sum(prop, index)
        out = tf.transpose(a=out, perm=perm)
        return out

    @staticmethod
    def compute_output_shape(input_shape: Sequence) -> tuple:
        """
        Compute output shape from input shape
        Args:
            input_shape (tuple/list): input shape list
        """
        prop_shape = input_shape[0]
        return prop_shape[0], None, prop_shape[-1]

    def get_config(self) -> dict:
        """
        Get layer configuration
        """
        config = {"alpha": self.alpha}
        base_config = super().get_config()
        config.update(base_config)
        return config  # type: ignore


class WeightedSet2Set(Set2Set):
    """
    This is a modified version from megnet.layers.readout.Set2Set.
    Here, in addition to taking features and indices as inputs, we also
    take a weight tensor. The input to the core logic is
    [features, weights, indices]
    """

    def build(self, input_shape: Sequence) -> None:
        """
        Build the output shape from input shapes
        Args:
            input_shape (tuple/list): input shape list
        """
        feature_shape, weight_shape, index_shape = input_shape
        super().build([feature_shape, index_shape])

    def call(self, inputs, mask=None):
        """
        Core logic of the layer

        Args:
            inputs (tuple): input tuple of length 3
            mask (tf.Tensor): not used here
        """
        features, weights, feature_graph_index = inputs
        feature_graph_index = tf.reshape(feature_graph_index, (-1,))
        _, _, count = tf.unique_with_counts(feature_graph_index)
        m = kb.dot(features, self.m_weight)
        if self.use_bias:
            m += self.m_bias

        self.h = tf.zeros(tf.stack([tf.shape(input=features)[0], tf.shape(input=count)[0], self.n_hidden]))
        self.c = tf.zeros(tf.stack([tf.shape(input=features)[0], tf.shape(input=count)[0], self.n_hidden]))
        q_star = tf.zeros(tf.stack([tf.shape(input=features)[0], tf.shape(input=count)[0], 2 * self.n_hidden]))
        for i in range(self.T):
            self.h, c = self._lstm(q_star, self.c)
            e_i_t = tf.reduce_sum(input_tensor=m * repeat_with_index(self.h, feature_graph_index), axis=-1)
            exp = tf.exp(e_i_t) * weights
            # print('exp shape ', exp.shape)
            seg_sum = tf.transpose(
                a=tf.math.segment_sum(tf.transpose(a=exp, perm=[1, 0]), feature_graph_index), perm=[1, 0]
            )
            seg_sum = tf.expand_dims(seg_sum, axis=-1)
            # print('seg_sum shape', seg_sum.shape)
            interm = repeat_with_index(seg_sum, feature_graph_index)
            # print('interm shape', interm.shape)
            a_i_t = exp / interm[..., 0]
            # print(a_i_t.shape)
            r_t = tf.transpose(
                a=tf.math.segment_sum(
                    tf.transpose(a=tf.multiply(m, a_i_t[:, :, None]), perm=[1, 0, 2]), feature_graph_index
                ),
                perm=[1, 0, 2],
            )
            q_star = kb.concatenate([self.h, r_t], axis=-1)
        return q_star
