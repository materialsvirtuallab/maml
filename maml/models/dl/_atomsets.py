"""
neural network models
"""
import math
from typing import Optional, Sequence, Union, List, Any

import numpy as np

from maml.base import KerasModel, BaseDescriber, BaseModel


def construct_atom_sets(
    input_dim: Optional[int] = None,
    is_embedding: bool = True,
    n_neurons: Sequence[int] = (64, 64),
    n_neurons_final: Sequence[int] = (64, 64),
    n_targets: int = 1,
    activation: str = "relu",
    embedding_vcal: int = 95,
    embedding_dim: int = 32,
    symmetry_func: Union[List[str], str] = "mean",
    optimizer: str = "adam",
    loss: str = "mse",
    compile_metrics: tuple = (),
    is_classification: bool = False,
    **symmetry_func_kwargs,
):
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
        optimizer (str): optimizer for the models
        loss (str): loss function for the models
        compile_metrics (tuple): metrics for validation
        symmetry_func_kwargs (dict): kwargs for symmetry function
    """
    from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate
    from tensorflow.keras.models import Model

    if is_embedding and input_dim is not None:
        raise ValueError("When embedding is used, input dim needs to be None")

    if is_embedding:
        inp = Input(shape=(None,), dtype="int32", name="node_id")
        out_ = Embedding(embedding_vcal, embedding_dim)(inp)
    else:
        inp = Input(shape=(None, input_dim), dtype="float32", name="node_feature_input")
        out_ = inp

    weight_inputs = Input(shape=(None,), dtype="float32", name="weight_input")
    node_ids = Input(shape=(None,), dtype="int32", name="node_in_graph_id")

    # start neural networks \phi
    for n_neuron in n_neurons:
        out_ = Dense(n_neuron, activation=activation)(out_)

    # apply symmetry function \rho
    if isinstance(symmetry_func, str):
        symmetry_func = [symmetry_func]

    symmetry_layers = []
    for symm in symmetry_func:
        if symm == "set2set":
            from maml.models.dl._layers import WeightedSet2Set

            layer = WeightedSet2Set(**symmetry_func_kwargs)
        elif symm == "mean":
            from maml.models.dl._layers import WeightedAverageLayer

            alpha = symmetry_func_kwargs.pop("alpha", 1)
            layer = WeightedAverageLayer(alpha=alpha)
        else:
            raise ValueError("symmetry function not supported")
        symmetry_layers.append(layer)
    outs = [i([out_, weight_inputs, node_ids]) for i in symmetry_layers]

    if len(outs) > 1:
        out_ = Concatenate(axis=-1)(outs)
    else:
        out_ = outs[0]

    # neural networks
    for n_neuron in n_neurons_final:
        out_ = Dense(n_neuron, activation=activation)(out_)

    if is_classification:
        final_act: Optional[str] = "sigmoid"
    else:
        final_act = None
    out_ = Dense(n_targets, activation=final_act)(out_)
    model = Model(inputs=[inp, weight_inputs, node_ids], outputs=out_)
    model.compile(optimizer, loss, metrics=compile_metrics)
    return model


class AtomSets(KerasModel):
    r"""
    This class implements the DeepSets models
    """

    def __init__(
        self,
        describer: Optional[BaseDescriber] = None,
        input_dim: Optional[int] = None,
        is_embedding: bool = True,
        n_neurons: Sequence[int] = (64, 64),
        n_neurons_final: Sequence[int] = (64, 64),
        n_targets: int = 1,
        activation: str = "relu",
        embedding_vcal: int = 95,
        embedding_dim: int = 32,
        symmetry_func: Union[List[str], str] = "mean",
        optimizer: str = "adam",
        loss: str = "mse",
        compile_metrics: tuple = (),
        is_classification: bool = False,
        **symmetry_func_kwargs,
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
            optimizer (str): optimizer for the models
            loss (str): loss function for the models
            symmetry_func_kwargs (dict): kwargs for symmetry function

        """
        input_dim = self.get_input_dim(describer, input_dim)
        model = construct_atom_sets(
            input_dim=input_dim,
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
            is_classification=is_classification,
            **symmetry_func_kwargs,
        )
        self.is_embedding = is_embedding
        super().__init__(model=model, describer=describer)

    def _get_data_generator(self, features, targets, batch_size=128, is_shuffle=True):
        if features is None:
            return None
        from tensorflow.keras.utils import Sequence as KerasSequence

        def _generate_atom_indices(lengths):
            max_length = max(lengths)
            res = np.tile(np.arange(max_length)[None, :], (len(lengths), 1))
            res2 = np.tile(np.arange(len(lengths))[:, None], (1, max_length))
            return res2[res < np.array(lengths)[:, None]]

        is_embedding = self.is_embedding

        class _DataGenerator(KerasSequence):
            def __init__(self, features=features, targets=targets, batch_size=batch_size, is_shuffle=is_shuffle):
                if isinstance(features[0], list) and len(features[0]) == 2:
                    self.features = [i[0] for i in features]
                    self.weights = [i[1] for i in features]
                else:
                    self.features = features
                    self.weights = [np.ones(shape=(len(i),)) for i in features]
                self.targets = targets
                self.batch_size = batch_size
                self.is_shuffle = is_shuffle

            def __len__(self):
                return math.ceil(len(self.features) / self.batch_size)

            def __getitem__(self, idx):
                features_temp = self.features[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_x = np.concatenate(features_temp, axis=0)
                if is_embedding:
                    batch_x = batch_x[..., 0]
                lengths = [len(i) for i in features_temp]
                f_index = _generate_atom_indices(lengths)
                batch_weights = self.weights[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]
                return (
                    np.array(batch_x)[None, ...],
                    np.concatenate(batch_weights)[None, :],
                    f_index[None, :],
                ), np.array(batch_y)[None, :]

            def on_epoch_end(self):
                """
                Codes executed at the end of each epoch
                """
                if self.is_shuffle:
                    indices = list(range(len(self.features)))
                    np.random.shuffle(indices)
                    self.features = [self.features[i] for i in indices]
                    self.weights = [self.weights[i] for i in indices]
                    self.targets = [self.targets[i] for i in indices]

        return _DataGenerator()

    def fit(
        self,
        features: Union[List, np.ndarray],
        targets: Union[List, np.ndarray] = None,
        val_features: Optional[Union[List, np.ndarray]] = None,
        val_targets: Optional[Union[List, np.ndarray]] = None,
        **kwargs,
    ) -> "BaseModel":
        """
        Args:
            features (list or np.ndarray): Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
            targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
            val_features (list or np.ndarray): validation features
            val_targets (list or np.ndarray): validation targets

        """
        batch_size = kwargs.pop("batch_size", 128)
        is_shuffle = kwargs.pop("is_shuffle", True)
        train_generator = self._get_data_generator(features, targets, batch_size=batch_size, is_shuffle=is_shuffle)
        val_generator = self._get_data_generator(
            val_features, val_targets, batch_size=batch_size, is_shuffle=is_shuffle
        )
        return self.model.fit(train_generator, validation_data=val_generator, **kwargs)  # type: ignore

    def _predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict the values given a set of inputs based on fitted models.

        Args:
            features (np.ndarray): array-like input features.

        Returns:
            List of output objects.
        """
        predict_generator = self._get_data_generator(features, [0] * len(features), is_shuffle=False, **kwargs)
        predicted = []
        for batch in predict_generator:
            predicted.append(self.model.predict(batch[0])[0])  # type: ignore
        return np.concatenate(predicted, axis=0)

    def evaluate(self, eval_objs: Any, eval_targets: Any, is_feature: bool = False):
        """
        Evaluate objs, targets

        Args:
            eval_objs (list): objs for evaluation
            eval_targets (list): target list for the corresponding objects
            is_feature (bool): whether x is feature matrix
        """
        eval_features = eval_objs if is_feature else self.describer.transform(eval_objs)
        eval_generator = self._get_data_generator(eval_features, eval_targets)
        return self.model.evaluate(eval_generator)
