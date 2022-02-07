import os
import unittest

import numpy as np
import pandas as pd
from pymatgen.util.testing import PymatgenTest

from maml.apps.symbolic import ISIS, SIS, L0BrutalForce
from maml.apps.symbolic._sis import _best_combination, _get_coeff

CWD = os.path.abspath(os.path.dirname(__file__))


class TestSIS(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        df = pd.read_csv(os.path.join(CWD, "sis_test.csv"), index_col=[0])
        cls.x = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x", "pow2", "exp"]]
        cls.y = df["y"]

    def test_sis(self):
        sis = SIS(gamma=0.5, selector=L0BrutalForce(1e-4))
        selected = sis.run(self.x.values, self.y)
        np.testing.assert_almost_equal(selected, [12, 10, 9])
        sis.set_gamma(0.1)
        self.assertEqual(sis.gamma, 0.1)
        sis.update_gamma(ratio=0.5)
        self.assertAlmostEqual(sis.gamma, 0.15)


class testISIS(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        df = pd.read_csv(os.path.join(CWD, "sis_test.csv"), index_col=[0])
        cls.x = df[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x", "pow2", "exp"]]
        cls.y = df["y"]

    def test_coeff(self):
        isis = ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-4)))
        coeff = _get_coeff(self.x.values, self.y)
        self.assertArrayAlmostEqual(coeff[10:], [1, 1, 1])

    def test_evaluate(self):
        isis = ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-4)), l0_regulate=False)
        _ = isis.run(self.x.values, self.y, max_p=4)
        mae = isis.evaluate(self.x.values, self.y)
        self.assertAlmostEqual(mae, 0)

    def test_best_combination(self):
        isis = ISIS(SIS(gamma=0.5, selector=L0BrutalForce(1e-4)), l0_regulate=False)
        _ = isis.run(self.x.values, self.y, max_p=4)
        comb_best, coeff_best, score_best = _best_combination(
            self.x.values, self.y, np.array([10, 11]), np.array([12, 0])
        )
        self.assertArrayEqual(comb_best, [10, 11, 12])
        self.assertArrayAlmostEqual(coeff_best, [1, 1, 1])
        self.assertAlmostEqual(score_best, 0)

    def test_isis(self):
        # Test update gamma
        np.random.seed(42)
        isis = ISIS(SIS(gamma=0.1, selector=L0BrutalForce(1e-4)))
        selected = isis.run(self.x.values, self.y, max_p=3)
        self.assertAlmostEqual(isis.sis.gamma, 0.1)
        np.testing.assert_equal(selected, [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
