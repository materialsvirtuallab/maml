"""
Test models
"""
from unittest import TestCase, main

from maml.models import MLP
from maml.describers import ElementStats


class TestAtomSets(TestCase):

    x = ["H2O", "Fe2O3"]
    y = [0.1, 0.2]
    model = MLP(describer=ElementStats.from_data("megnet_3"), n_neurons=(2, 2))

    def test_train(self):
        self.model.train(self.x, self.y, epochs=0)
        self.assertTrue(self.model.predict_objs(["H2O"]).shape == (1, 1))


if __name__ == "__main__":
    main()
