from maml.model.multi_species_nn import base_model, create_atomic_nn
from keras.layers import Input, Dense, Lambda, Add, Multiply
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
import unittest


class TestMultiSpeciesNN(unittest.TestCase):
    def test_create_atomic_nn(self):
        keras_input = Input(shape=(None, 3))
        keras_output = create_atomic_nn(keras_input, [3, 10, 1])
        model = Model(inputs=keras_input, outputs=keras_output)
        model.compile(loss='mse', optimizer=Adam(1e-2))
        x = np.array([[[0.1, 0.2, 0.3], [0.15, 0.36, 0.6]]])
        y = [1.0]
        model.fit(x, y, epochs=100, verbose=False)
        pred = model.predict(x)
        #print(pred)
        self.assertAlmostEqual(y[0], pred[0][0], 1)

    def test_base_model(self):
        model = base_model([3, 10, 1], ['A', 'B'], learning_rate=1e-2)
        # this simulates the case where there are 4 atom A and 6 atom B in
        # the structure and the energy per atom is 0.1
        features = [np.random.randn(1, 4, 3), np.random.randn(1, 6, 3)]
        outputs = [0.1]
        model.fit(features, outputs, epochs=100, verbose=0)
        pred = model.predict(features)
        self.assertAlmostEqual(outputs[0], pred[0][0], 1)


if __name__ == "__main__":
    unittest.main()
