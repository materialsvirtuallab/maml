"""
Tests for preprocessing
"""
from __future__ import annotations

import unittest

from pymatgen.core import Lattice, Structure

from maml.utils import DummyScaler, StandardScaler


class TestPreprocessing(unittest.TestCase):
    def test_ss(self):
        ss = StandardScaler(mean=0, std=10, is_intensive=False)
        self.assertAlmostEqual(ss.transform(10, n=10), 0.1)

        s = Structure(Lattice.cubic(3.16), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])

        ss2 = StandardScaler.from_training_data(structures=[s, s], targets=[0, 0], is_intensive=True)
        self.assertAlmostEqual(ss2.transform(10, n=10), 10)
        self.assertAlmostEqual(ss2.inverse_transform(10, n=10), 10)

    def test_dummy(self):
        dummy = DummyScaler()
        self.assertAlmostEqual(dummy.transform(100), 100)
        self.assertAlmostEqual(DummyScaler.from_training_data([0, 0], [0, 0]).inverse_transform(100), 100)


if __name__ == "__main__":
    unittest.main()
