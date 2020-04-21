import unittest
import os

import numpy as np
import json

from maml.apps.symbolic import SIS, DantzigSelector

CWD = os.path.abspath(os.path.dirname(__file__))


class TestSIS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        with open(os.path.join(CWD, 'test_data_sis.json'), 'r') as f:
            djson = json.load(f)
        cls.x = np.array(djson['x'])
        cls.beta = np.array(djson['beta'])
        cls.y = cls.x.dot(cls.beta)
        cls.lasso_alpha = 0.1

    def test_sis(self):
        sis = SIS(gamma=0.1, selector=DantzigSelector(0.1))
        selected = sis.run(self.x, self.y)
        np.testing.assert_almost_equal(selected, [91, 59, 65, 84, 15, 23, 89,  1, 76])


if __name__ == "__main__":
    unittest.main()
