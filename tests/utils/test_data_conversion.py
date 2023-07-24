# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np
from monty.serialization import loadfn

from maml.utils._data_conversion import convert_docs, pool_from

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.dirname(__file__), "../apps/pes/datapool.json"))


class PorcessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(CWD)
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.test_pool = test_datapool
        self.test_structures = []
        self.test_energies = []
        self.test_forces = []
        self.test_stresses = []
        for d in self.test_pool:
            self.test_structures.append(d["structure"])
            self.test_energies.append(d["outputs"]["energy"])
            self.test_forces.append(d["outputs"]["forces"])
            self.test_stresses.append(d["outputs"]["virial_stress"])

    def test_pool_from(self):
        test_pool = pool_from(self.test_structures, self.test_energies, self.test_forces, self.test_stresses)
        for p1, p2 in zip(test_pool, self.test_pool):
            assert p1["outputs"]["energy"] == p2["outputs"]["energy"]
            assert p1["outputs"]["forces"] == p2["outputs"]["forces"]
            assert p1["outputs"]["virial_stress"] == p2["outputs"]["virial_stress"]

    def test_convert_docs(self):
        _, df = convert_docs(self.test_pool, include_stress=False)
        test_energies = df[df["dtype"] == "energy"]["y_orig"]
        assert not np.any(test_energies - self.test_energies)
        test_forces = df[df["dtype"] == "force"]["y_orig"]
        for force1, force2 in zip(test_forces, np.array(self.test_forces).ravel()):
            assert force1 == force2

        _, df = convert_docs(self.test_pool, include_stress=True)
        test_energies = df[df["dtype"] == "energy"]["y_orig"]
        assert not np.any(test_energies - self.test_energies)
        test_forces = df[df["dtype"] == "force"]["y_orig"]
        for force1, force2 in zip(test_forces, np.array(self.test_forces).ravel()):
            assert force1 == force2
        test_stresses = df[df["dtype"] == "stress"]["y_orig"]
        for stress1, stress2 in zip(test_stresses, np.array(self.test_stresses).ravel()):
            assert stress1 == stress2


if __name__ == "__main__":
    unittest.main()
