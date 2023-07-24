"""
Test models
"""
from __future__ import annotations

from unittest import TestCase, main

import numpy as np

from maml.describers import SiteElementProperty
from maml.models import AtomSets


class TestAtomSets(TestCase):
    x = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int32).reshape((1, -1))
    x_vec = np.random.normal(size=(1, 6, 20))
    indices = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int32).reshape((1, -1))
    y = np.array([[0.1, 0.2]]).reshape((1, 2, 1))

    model1 = AtomSets(
        describer=SiteElementProperty(),
        is_embedding=True,
        symmetry_func="mean",
        n_neurons=(8, 8),
        n_neurons_final=(4, 4),
        n_targets=1,
    )
    model2 = AtomSets(
        input_dim=20,
        is_embedding=False,
        symmetry_func="set2set",
        n_neurons=(4, 4),
        n_neurons_final=(4, 4),
        T=2,
        n_hidden=10,
    )

    def test_predict(self):
        res = self.model1.predict_objs(["H2O", "FeO"])
        print(res.shape, " res.shape")
        assert res.shape == (2, 1)
        res3 = self.model2.model.predict([self.x_vec, np.ones_like(self.indices), self.indices])
        assert res3.shape == (1, 2, 1)


if __name__ == "__main__":
    main()
