"""
neural network models
"""
import math
from typing import Optional, Sequence, Union, List

import numpy as np

from maml.base import KerasModel, BaseDescriber, BaseModel


def construct_deep_sets(
        input_dim: Optional[int] = None,
        is_embedding: bool = True,
        n_neurons: Sequence[int] = (64, 64),
        n_neurons_final: Sequence[int] = (64, 64),
        n_targets: int = 1,
        activation: str = 'relu',
        embedding_vcal: int = 95,
        embedding_dim: int = 32,
        symmetry_func: str = "mean",
        optimizer: str = 'adam',
        loss: str = 'mse',
        compile_metrics: tuple = (),
        **symmetry_func_kwargs):
    r"""
    f(X) = \rho(\sum_{x \in X} \phi(x)), where X is a set.
    \phi is implemented as a neural network and \rho is a symmetry function.

    todo: implement attention mechanism

    Args:
        input_dim (int): input dimension, if None, then integer inputs + embedding are assumed.
        is_embedding (bool): whether the input should be embedded
        n_neurons (tuple): number of hidden-layer neurons before passing to symmetry function
        n_neurons_final (tuple): number of hidden-layer neurons after symmetry function
        n_targets (int): number of output targets
        activation (str): activation function
        embedding_vcal (int): int, embedding vocabulary
        embedding_dim (int): int, embedding dimension
        symmetry_func (str): symmetry function, choose from ['set2set', 'sum', 'mean',
            'max', 'min', 'prod']
        optimizer (str): optimizer for the model
        loss (str): loss function for the model
        symmetry_func_kwargs (dict): kwargs for symmetry function
    """
    from tensorflow.keras.layers import Input, Dense, Embedding
    from tensorflow.keras.models import Model
    if is_embedding and input_dim is not None:
        raise ValueError("When embedding is used, input dim needs to be None")

    if is_embedding:
        inp = Input(shape=(None, ), dtype='int32', name='node_id')
        out_ = Embedding(embedding_vcal, embedding_dim)(inp)
    else:
        inp = Input(shape=(None, input_dim),
                    dtype='float32', name='node_feature_input')
        out_ = inp
    node_ids = Input(shape=(None,), dtype='int32', name='node_in_graph_id')

    # start neural networks \phi
    for n_neuron in n_neurons:
        out_ = Dense(n_neuron, activation=activation)(out_)

    # apply symmetry function \rho
    if symmetry_func == 'set2set':
        from megnet.layers import Set2Set
        layer = Set2Set(**symmetry_func_kwargs)
    elif symmetry_func in ['mean', 'sum', 'max', 'min', 'prod']:
        from megnet.layers import LinearWithIndex
        layer = LinearWithIndex(mode=symmetry_func)
    else:
        raise ValueError("symmetry function not supported")

    out_ = layer([out_, node_ids])

    # neural networks
    for n_neuron in n_neurons_final:
        out_ = Dense(n_neuron, activation=activation)(out_)

    out_ = Dense(n_targets)(out_)
    model = Model(inputs=[inp, node_ids], outputs=out_)
    model.compile(optimizer, loss, metrics=compile_metrics)
    return model


class DeepSets(KerasModel):
    r"""
    This class implements the DeepSets model
    """

    def __init__(self,
                 describer: Optional[BaseDescriber] = None,
                 input_dim: Optional[int] = None,
                 is_embedding: bool = True,
                 n_neurons: Sequence[int] = (64, 64),
                 n_neurons_final: Sequence[int] = (64, 64),
                 n_targets: int = 1,
                 activation: str = 'relu',
                 embedding_vcal: int = 95,
                 embedding_dim: int = 32,
                 symmetry_func: str = "mean",
                 optimizer: str = 'adam',
                 loss: str = 'mse',
                 compile_metrics: tuple = (),
                 **symmetry_func_kwargs
                 ):
        """
        Args:
            describer (BaseDescriber): site describers
            input_dim (int): input dimension, if None, then integer inputs + embedding are assumed.
            is_embedding (bool): whether the input should be embedded
            n_neurons (tuple): number of hidden-layer neurons before passing to symmetry function
            n_neurons_final (tuple): number of hidden-layer neurons after symmetry function
            n_targets (int): number of output targets
            activation (str): activation function
            embedding_vcal (int): int, embedding vocabulary
            embedding_dim (int): int, embedding dimension
            symmetry_func (str): symmetry function, choose from ['set2set', 'sum', 'mean',
                'max', 'min', 'prod']
            optimizer (str): optimizer for the model
            loss (str): loss function for the model
            symmetry_func_kwargs (dict): kwargs for symmetry function

        """
        model = construct_deep_sets(input_dim=input_dim,
                                    is_embedding=is_embedding,
                                    n_neurons=n_neurons,
                                    n_neurons_final=n_neurons_final,
                                    n_targets=n_targets,
                                    activation=activation,
                                    embedding_vcal=embedding_vcal,
                                    embedding_dim=embedding_dim,
                                    symmetry_func=symmetry_func,
                                    optimizer=optimizer,
                                    loss=loss,
                                    compile_metrics=compile_metrics,
                                    **symmetry_func_kwargs)
        self.is_embedding = is_embedding
        super().__init__(model=model, describer=describer)

    def _get_data_generator(self, features, targets, batch_size=128, is_shuffle=True):
        from tensorflow.keras.utils import Sequence as KerasSequence

        def _generate_atom_indices(lengths):
            max_length = max(lengths)
            res = np.tile(np.arange(max_length)[None, :], (len(lengths), 1))
            res2 = np.tile(np.arange(len(lengths))[:, None], (1, max_length))
            return res2[res < np.array(lengths)[:, None]]

        is_embedding = self.is_embedding

        class _DataGenerator(KerasSequence):
            def __init__(self, features=features, targets=targets, batch_size=batch_size,
                         is_shuffle=is_shuffle):
                self.features, self.targets = features, targets
                self.batch_size = batch_size
                self.is_shuffle = is_shuffle

            def __len__(self):
                return math.ceil(len(self.features) / self.batch_size)

            def __getitem__(self, idx):
                features_temp = self.features[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_x = np.concatenate(features_temp, axis=0)
                if is_embedding:
                    batch_x = batch_x[..., 0]
                lengths = [len(i) for i in features_temp]
                f_index = _generate_atom_indices(lengths)
                batch_y = self.targets[idx * self.batch_size: (idx + 1) * self.batch_size]
                return (np.array(batch_x)[None, :], f_index[None, :]), np.array(batch_y)[None, :]

            def on_epoch_end(self):
                if self.is_shuffle:
                    indices = list(range(len(self.features)))
                    np.random.shuffle(indices)
                    self.features = [self.features[i] for i in indices]
                    self.targets = [self.targets[i] for i in indices]
        return _DataGenerator()

    def fit(self, features: Union[List, np.ndarray],
            targets: Union[List, np.ndarray] = None, **kwargs) -> "BaseModel":
        """
        Args:
            features (list or np.ndarray): Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
            targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
        """
        batch_size = kwargs.pop('batch_size', 128)
        is_shuffle = kwargs.pop('is_shuffle', True)
        train_generator = self._get_data_generator(features, targets, batch_size=batch_size, is_shuffle=is_shuffle)
        if 'val_features' in kwargs and 'val_targets' in kwargs:
            val_generator = self._get_data_generator(kwargs.pop('val_features'),
                                                     kwargs.pop('val_targets'),
                                                     batch_size=batch_size, is_shuffle=is_shuffle)
        else:
            val_generator = None

        return self.model.fit(train_generator,  # type: ignore
                              validation_data=val_generator, **kwargs)

    def _predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict the values given a set of inputs based on fitted model.

        Args:
            features (np.ndarray): array-like input features.

        Returns:
            List of output objects.
        """
        predict_generator = self._get_data_generator(features, [0] * len(features),
                                                     is_shuffle=False, **kwargs)
        predicted = []
        for batch in predict_generator:
            predicted.append(self.model.predict(batch[0])[0])  # type: ignore
        return np.concatenate(predicted, axis=0)
