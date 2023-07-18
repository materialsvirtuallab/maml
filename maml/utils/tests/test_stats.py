from __future__ import annotations

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
        self.assertAlmostEqual(Stats.mean(self.x, self.weights), np.average(self.x, weights=self.weights))
        self.assertAlmostEqual(Stats.inverse_mean(self.x), np.mean(1.0 / np.array(self.x)))

        self.assertAlmostEqual(Stats.mean(self.x, self.weights), Stats.average(self.x, self.weights))

    def test_max_min_range(self):
        assert Stats.max(self.x) == 5
        assert Stats.min(self.x) == 1
        assert Stats.range(self.x) == 4

    def test_moment(self):
        np.testing.assert_almost_equal(Stats.moment(self.x, max_order=2), [np.mean(self.x), np.std(self.x)])
        self.assertAlmostEqual(Stats.moment(self.x, order=1), np.mean(self.x))

        self.assertAlmostEqual(Stats.moment(self.x), np.mean(self.x))

    def test_kurtosis_skew(self):
        self.assertAlmostEqual(Stats.skewness(self.x), skew(self.x))
        self.assertAlmostEqual(Stats.kurtosis(self.x), kurtosis(self.x, fisher=False))

    def test_geometric_mean(self):
        self.assertAlmostEqual(
            Stats.geometric_mean(self.x, self.weights),
            np.prod(np.power(self.x, self.weights)) ** (1.0 / np.sum(self.weights)),
        )

    def test_shifted_geometric_mean(self):
        self.assertAlmostEqual(Stats.shifted_geometric_mean([0.12, -0.08, 0.02], shift=1), 0.0167215)

    def test_harmonic_mean(self):
        self.assertAlmostEqual(
            Stats.harmonic_mean(self.x, self.weights),
            np.sum(self.weights) / np.sum(np.array(self.weights) / np.array(self.x)),
        )

    def test_allowed_stats(self):
        assert "best" not in Stats.allowed_stats
        assert "average" in Stats.allowed_stats

    def test_power_mean(self):
        p1 = Stats.power_mean(self.x, self.weights, p=0)
        self.assertAlmostEqual(p1, Stats.geometric_mean(self.x, self.weights))
        p2 = Stats.power_mean(self.x, self.weights, p=2)

        holder_mean_2 = np.sum(np.array(self.weights) * np.array(self.x) ** 2) ** (1.0 / 2)
        self.assertAlmostEqual(p2, holder_mean_2)

    def test_mode(self):
        self.assertAlmostEqual(Stats.mode(self.x), 1.5)
        self.assertAlmostEqual(Stats.mode(self.x, self.weights), 2)

    def test_mae(self):
        self.assertAlmostEqual(
            Stats.mean_absolute_deviation(self.x), np.mean(np.abs(np.array(self.x) - np.mean(self.x)))
        )


if __name__ == "__main__":
    unittest.main()
