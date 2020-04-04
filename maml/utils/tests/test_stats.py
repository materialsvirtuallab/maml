import unittest

import numpy as np
from scipy.stats import kurtosis, skew

from maml.utils import Stats


class TestStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = [1, 1, 2, 3, 4, 5, 2]
        cls.weights = [0.1, 0.2, 0.3, 0.1, 0.2, 0.05, 0.05]

    def test_mean(self):
        self.assertAlmostEqual(Stats.mean(self.x), np.mean(self.x))
        self.assertAlmostEqual(Stats.mean(self.x, self.weights),
                               np.average(self.x, weights=self.weights))
        self.assertAlmostEqual(Stats.mean(self.x, self.weights),
                               Stats.average(self.x, self.weights))

    def test_max_min_range(self):
        self.assertEqual(Stats.max(self.x), 5)
        self.assertEqual(Stats.min(self.x), 1)
        self.assertEqual(Stats.range(self.x), 4)

    def test_moment(self):
        np.testing.assert_almost_equal(Stats.moment(self.x, max_order=2),
                                       [np.mean(self.x), np.std(self.x)])
        self.assertAlmostEqual(Stats.moment(self.x, order=1),
                               np.mean(self.x))

        self.assertAlmostEqual(Stats.moment(self.x),
                               np.mean(self.x))

    def test_kurtosis_skew(self):
        self.assertAlmostEqual(Stats.skewness(self.x), skew(self.x))
        self.assertAlmostEqual(Stats.kurtosis(self.x), kurtosis(self.x, fisher=False))

    def test_geometric_mean(self):
        self.assertAlmostEqual(Stats.geometric_mean(self.x, self.weights),
                               np.prod(np.power(self.x, self.weights))**(1./np.sum(self.weights)))

    def test_harmonic_mean(self):
        self.assertAlmostEqual(Stats.harmonic_mean(self.x, self.weights),
                               np.sum(self.weights) / np.sum(np.array(self.weights) / np.array(self.x)))

    def test_allowed_stats(self):
        self.assertNotIn("best",  Stats.allowed_stats)
        self.assertIn("average", Stats.allowed_stats)


if __name__ == "__main__":
    unittest.main()
