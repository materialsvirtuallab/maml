import unittest
import os

import numpy as np
import pandas as pd

from maml.apps.symbolic import SIS, ISIS, L0BrutalForce

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
        sis.set_gamma(0.1)
        self.assertEqual(sis.gamma, 0.1)
        sis.update_gamma(step=0.5)
        self.assertAlmostEqual(sis.gamma, 0.15)

    def test_isis(self):
        isis = ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-4)))
        selected = isis.run(self.x.values, self.y, max_p=10)
        np.testing.assert_equal(selected, [10, 11, 12,  4,  5,  6,  0,  3,  2,  8])

        isis = ISIS(SIS(gamma=0.1, selector=L0BrutalForce(1e-4)))
        selected = isis.run(self.x.values, self.y, max_p=10)
        self.assertEqual(isis.sis.gamma, 0.3375)
        np.testing.assert_equal(selected, [10, 11, 12,  4,  5,  6,  0,  3,  2,  8])

if __name__ == "__main__":
    unittest.main()