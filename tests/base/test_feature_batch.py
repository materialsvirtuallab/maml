from __future__ import annotations

import unittest

import numpy as np

from maml.base import get_feature_batch
from maml.base._feature_batch import stack_first_dim


class TestFeatureBatch(unittest.TestCase):
    def test_get_feature_batch(self):
        x = [np.random.normal(size=(10, 20)), np.random.normal(size=(10, 20)), np.random.normal(size=(10, 20))]
        y = get_feature_batch("stack_first_dim")(x)
        assert y.shape == (3, 10, 20)
        y = get_feature_batch(stack_first_dim)(x)
        assert y.shape == (3, 10, 20)
        self.assertRaises(KeyError, get_feature_batch, fb_name="unknown_batch")

        y = get_feature_batch()(x)
        np.testing.assert_almost_equal(y[0], x[0])


if __name__ == "__main__":
    unittest.main()
