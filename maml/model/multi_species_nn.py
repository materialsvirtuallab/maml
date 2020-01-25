from keras.layers import Input, Dense, Lambda, Add, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


def create_atomic_nn(keras_input, layers):
    """
    Create a basic multi-layer perceptron model for each specie and output the sum of energies for all atoms
    of that specie.
    In the structure, the features of one specie are a [nb_atom, feature_dim] matrix,
    the output of this function is a number, i.e., the total energy.
    Therefore, the keras_input argument should have a shape of [None, feature_dim].

    :param keras_input: Keras layer, the input dimension is [None, feature_dim], where the None dimension is the nb of
        atoms
    :param layers: list, layer configuration, e.g., [30, 31, 32, 1] indicates
    that the input dimension is 30, with two hidden layers having 31 and 32 units and 1 output
    :return: Keras layer, layer object for latter steps
    """
    for i in range(len(layers) - 2):
        keras_input = Dense(layers[i + 1], activation='relu')(keras_input)
    keras_input = Dense(layers[-1])(keras_input)
    keras_output = Lambda(lambda x: K.sum(x, axis=1), output_shape=(1,))(keras_input)
    return keras_output


def base_model(layers, species, learning_rate=1e-3):
    """
    Build a multi-specie model that predicts the energy per atom
    Since the we cannot guarantee that all structures will have the same number of atoms
    nor same number of atoms in the specific specie type, the default input x should have the dimension
    [(1, nb_specie_1, feature_dim), (1, nb_specie_2, feature_dim), ..., (1, nb_specie_n, feature_dim)]
    Essentially, the fit step takes one structure-energy/atom pair at a time.

    One can also group the same structure types (same nb of atoms in each specie category) together
    so that the x has the dimension
    [(nb_structure, nb_specie_1, feature_dim), (nb_structure, nb_specie_2, feature_dim), ...,
    (nb_structure, nb_specie_n, feature_dim)], in this case the target y for fitting is a numpy array having
    dimension (nb_structure, 1)

    :param layers: list, number of neurons in each layer for all the species
    :param species: list, list of species list
    :param learning_rate: float, learning rate for Adam optimizer
    :return: keras model
    """
    outputs = []
    inputs = []
    atom_nums = []
    for i, _ in enumerate(species):
        # create input layer for each specie
        keras_input = Input(shape=(None, layers[0]))
        # calculate the atom number for each specie
        atom_nums.append(Lambda(lambda x: K.cast(K.shape(x)[1], dtype='float'),
                                output_shape=(1,))(keras_input))
        inputs.append(keras_input)
        # create the output for each specie category
        outputs.append(create_atomic_nn(keras_input, layers))
    # calculate the total energy by adding the specie total energy
    outputs = Add()(outputs)
    # calculate the total number of atoms
    total_num_atom = Add()(atom_nums)
    # calculate the energy per atom
    num_inv = Lambda(lambda x: 1. / x, output_shape=(1,))(total_num_atom)
    outputs = Multiply()([outputs, num_inv])
    # construct the keras model
    model = Model(inputs=inputs, outputs=outputs)
    # compile model with mean squared error as loss function and Adam as optimizer
    model.compile(loss='mse', optimizer=Adam(learning_rate))
    return model
