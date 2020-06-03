"""
Test models
"""
from unittest import TestCase, main

from maml.model import MLP
from maml.describer import ElementStats


class TestAtomSets(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.x = ['H2O', 'Fe2O3']
        cls.y = [0.1, 0.2]
        cls.model = MLP(describer=ElementStats.from_data("megnet_3"))

    def test_train(self):
        self.model.train(self.x, self.y, epochs=1)
        self.assertTrue(self.model.predict_objs(['H2O']).shape == (1, 1))


if __name__ == "__main__":
    main()
