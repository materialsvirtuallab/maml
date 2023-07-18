from __future__ import annotations

import os
import unittest

import numpy as np
from pymatgen.core.structure import Structure

from maml.apps.bowsr.acquisition import ensure_rng
from maml.apps.bowsr.perturbation import WyckoffPerturbation
from maml.apps.bowsr.preprocessing import DummyScaler, StandardScaler
from maml.apps.bowsr.target_space import TargetSpace

test_lfpo = Structure.from_file(os.path.join(os.path.dirname(__file__), "test_structures", "test_lfpo.cif"))


class TargetSpaceTest(unittest.TestCase):
    def setUp(self):
        self.wyckoff_dims = [2, 2, 3]
        self.wps = [WyckoffPerturbation(62, "c"), WyckoffPerturbation(62, "c"), WyckoffPerturbation(62, "d")]
        self.wps[0].sanity_check(test_lfpo[4])
        self.wps[1].sanity_check(test_lfpo[12])
        self.wps[2].sanity_check(test_lfpo[20])
        self.abc_dim = 3
        self.angles_dim = 0
        self.space1 = TargetSpace(
            target_func=lambda x: sum(x**2),
            wps=self.wps,
            abc_dim=self.abc_dim,
            angles_dim=self.angles_dim,
            relax_coords=True,
            relax_lattice=True,
            scaler=DummyScaler(),
            random_state=ensure_rng(42),
        )
        self.space2 = TargetSpace(
            target_func=lambda x: sum(x * 2),
            wps=self.wps,
            abc_dim=self.abc_dim,
            angles_dim=self.angles_dim,
            relax_coords=True,
            relax_lattice=False,
            scaler=DummyScaler(),
            random_state=ensure_rng(42),
        )
        self.space3 = TargetSpace(
            target_func=lambda x: np.mean(x * 3),
            wps=self.wps,
            abc_dim=self.abc_dim,
            angles_dim=self.angles_dim,
            relax_coords=True,
            relax_lattice=True,
            scaler=DummyScaler(),
            random_state=ensure_rng(42),
        )
        self.space4 = TargetSpace(
            target_func=lambda x: np.mean(x * 3),
            wps=self.wps,
            abc_dim=self.abc_dim,
            angles_dim=self.angles_dim,
            relax_coords=True,
            relax_lattice=True,
            scaler=StandardScaler(),
            random_state=ensure_rng(42),
        )

    def test_attributes(self):
        self.space1.set_bounds()
        assert [wp.dim for wp in self.space1.wps] == self.wyckoff_dims
        assert self.space1.abc_dim == self.abc_dim
        assert self.space1.angles_dim == self.angles_dim
        assert self.space1.relax_lattice
        assert len(self.space1.bounds) == self.space1.dim
        assert not np.all(abs(self.space1.bounds) <= 0.2)
        assert self.space1.dim == 10
        assert len(self.space1) == 0
        assert isinstance(self.space1.scaler, DummyScaler)
        self.space2.set_bounds()
        assert not self.space2.relax_lattice
        assert len(self.space2.bounds) == self.space2.dim
        assert np.all(abs(self.space2.bounds) <= 0.2)
        assert self.space2.dim == 7
        assert len(self.space2) == 0
        assert isinstance(self.space2.scaler, DummyScaler)
        self.space2.set_bounds(element_wise_wyckoff_bounds={"Fe": 1.0})
        assert not np.any(abs(self.space2.bounds) <= 0.1)
        self.space3.set_bounds()
        assert self.space3.relax_lattice
        assert len(self.space3.bounds) == self.space3.dim
        assert not np.any(abs(self.space3.bounds) >= 20)
        assert self.space3.dim == 10
        assert len(self.space3) == 0
        assert isinstance(self.space3.scaler, DummyScaler)
        self.space3.set_bounds(abc_bound=15)
        assert np.all(abs(self.space3.bounds) <= 15)
        assert self.space3.dim == 10
        assert len(self.space3) == 0
        self.space4.set_bounds()
        assert [wp.dim for wp in self.space4.wps] == self.wyckoff_dims
        assert self.space4.abc_dim == self.abc_dim
        assert self.space4.angles_dim == self.angles_dim
        assert self.space4.relax_lattice
        assert len(self.space4.bounds) == self.space4.dim
        assert not np.all(abs(self.space4.bounds) <= 0.1)
        assert self.space4.dim == 10
        assert len(self.space4) == 0
        assert isinstance(self.space4.scaler, StandardScaler)

    def test_sample(self):
        self.space1.set_bounds()
        self.space2.set_bounds()
        self.space3.set_bounds()
        self.space4.set_bounds()

        for _ in range(10):
            sample = self.space1.uniform_sample()
            assert len(sample) == self.space1.dim
            assert not np.any(abs(sample) > 1.2)
        for _ in range(10):
            sample = self.space2.uniform_sample()
            assert len(sample) == self.space2.dim
            assert np.all(abs(sample) <= 0.2)
        for _ in range(10):
            sample = self.space3.uniform_sample()
            assert len(sample) == self.space3.dim
            assert -20 <= sample[-1] <= 20
        for _ in range(10):
            sample = self.space4.uniform_sample()
            assert len(sample) == self.space4.dim
            assert not np.any(abs(sample) > 1.2)

    def test_register(self):
        self.space1.set_bounds()
        self.space2.set_bounds()
        self.space3.set_bounds()
        self.space4.set_bounds()
        mean = np.random.random(self.space4.dim)
        std = np.random.random(self.space4.dim)
        self.space4.scaler = StandardScaler(mean=mean, std=std)

        for _ in range(10):
            sample = self.space1.uniform_sample()
            target = self.space1.target_func(sample)
            self.space1.register(sample, target)
        assert len(self.space1) == 10
        assert all(abs(np.sum(self.space1.params**2, axis=1) - self.space1.target) < 0.01)

        for _ in range(20):
            sample = self.space2.uniform_sample()
            target = self.space2.target_func(sample)
            self.space2.register(sample, target)
        assert len(self.space2) == 20
        assert all(abs(np.sum(self.space2.params * 2, axis=1) - self.space2.target) < 0.01)

        for _ in range(15):
            sample = self.space3.uniform_sample()
            target = self.space3.target_func(sample)
            self.space3.register(sample, target)
        assert len(self.space3) == 15
        assert all(abs(np.mean(self.space3.params * 3, axis=1) - self.space3.target) < 0.01)

        for _ in range(10):
            sample = self.space4.uniform_sample()
            target = self.space4.target_func(sample)
            self.space4.register(sample, target)
        assert len(self.space4) == 10
        assert all(
            abs(np.mean(self.space4.scaler.inverse_transform(self.space4.params) * 3, axis=1) - self.space4.target)
            < 0.01
        )

    def test_probe(self):
        self.space1.set_bounds()
        self.space2.set_bounds()
        self.space3.set_bounds()
        self.space4.set_bounds()
        mean = np.random.random(self.space4.dim)
        std = np.random.random(self.space4.dim)
        self.space4.scaler = StandardScaler(mean=mean, std=std)

        for i in np.arange(0, 10, 2):
            sample = self.space1.uniform_sample()
            target1 = self.space1.target_func(sample)
            self.space1.register(sample, target1)
            assert len(self.space1) == i + 1
            target2 = self.space1.probe(sample)
            assert len(self.space1) == i + 2
            self.assertAlmostEqual(target1, target2)

        for i in np.arange(0, 20, 2):
            sample = self.space2.uniform_sample()
            target1 = self.space2.target_func(sample)
            self.space2.register(sample, target1)
            assert len(self.space2) == i + 1
            target2 = self.space2.probe(sample)
            assert len(self.space2) == i + 2
            self.assertAlmostEqual(target1, target2)

        for i in range(0, 20, 2):
            sample = self.space3.uniform_sample()
            target1 = self.space3.target_func(sample)
            self.space3.register(sample, target1)
            assert len(self.space3) == i + 1
            target2 = self.space3.probe(sample)
            assert len(self.space3) == i + 2
            self.assertAlmostEqual(target1, target2)

        for i in range(0, 10, 2):
            sample = self.space4.uniform_sample()
            target1 = self.space4.target_func(sample)
            self.space4.register(sample, target1)
            assert len(self.space4) == i + 1
            target2 = self.space4.probe(sample)
            assert len(self.space4) == i + 2
            self.assertAlmostEqual(target1, target2)

    def test_set_empty(self):
        self.space1.set_bounds()
        self.space2.set_bounds()
        self.space3.set_bounds()
        self.space4.set_bounds()
        mean = np.random.random(self.space4.dim)
        std = np.random.random(self.space4.dim)
        self.space4.scaler = StandardScaler(mean=mean, std=std)

        for i in range(10):
            sample = self.space1.uniform_sample()
            self.space1.probe(sample)
            assert len(self.space1) == i + 1
        self.space1.set_empty()
        assert len(self.space1) == 0

        for i in range(20):
            sample = self.space2.uniform_sample()
            self.space2.probe(sample)
            assert len(self.space2) == i + 1
        self.space2.set_empty()
        assert len(self.space2) == 0

        for i in range(15):
            sample = self.space3.uniform_sample()
            self.space3.probe(sample)
            assert len(self.space3) == i + 1
        self.space3.set_empty()
        assert len(self.space3) == 0

        for i in range(10):
            sample = self.space4.uniform_sample()
            self.space4.probe(sample)
            assert len(self.space4) == i + 1
        self.space4.set_empty()
        assert len(self.space4) == 0


if __name__ == "__main__":
    unittest.main()
