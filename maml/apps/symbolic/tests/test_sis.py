import unittest
import os

import numpy as np
import pandas as pd

from maml.apps.symbolic import SIS, L0BrutalForce

CWD = os.path.abspath(os.path.dirname(__file__))


class TestSIS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        df = pd.read_csv(os.path.join(CWD, "sis_test.csv"), index_col=[0])
        cls.x = df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', \
                    'x7', 'x8', 'x9', 'x', 'pow2', 'exp']]
        cls.y = df['y']

    def test_sis(self):
        sis = SIS(gamma=0.5, selector=L0BrutalForce(1e-4))
        selected = sis.run(self.x.values, self.y)
        np.testing.assert_almost_equal(selected, [12, 10, 9])

    def test_isis(self):
        sis = SIS(gamma=0.5, selector=L0BrutalForce(1e-4))
        selected = sis.isis(self.x.values, self.y, d=12)
        np.testing.assert_equal(selected, [10, 11, 12, 0, 1, 2, 3, 4, 5, 8, 9, 6])


if __name__ == "__main__":
    unittest.main()
