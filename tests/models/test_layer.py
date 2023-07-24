from __future__ import annotations

import unittest

import numpy as np

from maml.models.dl import WeightedAverageLayer, WeightedSet2Set


class TestLayer(unittest.TestCase):
    def test_weighted_average(self):
        alpha = 0.4
        wal = WeightedAverageLayer(alpha=alpha)
        x = np.random.normal(size=(1, 5, 2))
        weights = np.array([0.1, 0.2, 0.1, 0.4, 0.5]).reshape((1, -1))
        indices = np.array([0, 0, 0, 1, 1]).reshape((1, -1))
        out = wal.call([x, weights, indices])

        x = x[0, :3, :]
        weights = weights[0, :3]
        exp_weights = weights**alpha
        out_np = np.sum(x * exp_weights[:, None], axis=0) / np.sum(exp_weights)
        np.testing.assert_array_almost_equal(out.numpy()[0][0], out_np)
        assert wal.get_config().get("alpha") == alpha
        assert wal.compute_output_shape([[1, 5, 2], [1, 5], [1, 5]]) == (1, None, 2)

    def test_weighted_set2set(self):
        maml_set = WeightedSet2Set(T=2, n_hidden=2)
        maml_set.build([[1, 3, 2], [1, 3], [1, 3]])
        x = np.random.normal(size=(1, 3, 2))
        weights = np.array([0.5, 0.5, 1], dtype=np.float64).reshape((1, -1))
        indices = np.array([0, 0, 0], dtype="int32").reshape((1, -1))
        res = maml_set([x, weights, indices])
        assert res.shape == (1, 1, 4)


if __name__ == "__main__":
    unittest.main()
