"""
neural network models
"""
from typing import Optional, Sequence

from maml.base import KerasModel, BaseDescriber


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
    model.compile(optimizer, loss)
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
                                    **symmetry_func_kwargs)
        super().__init__(model=model, describer=describer)
