# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import os
import shutil
import tempfile
import unittest

import numpy as np
from monty.os.path import which
from monty.serialization import loadfn
from pymatgen.core import Structure

from maml.apps.pes import GAPotential

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.abspath(os.path.dirname(__file__)), "datapool.json"))


class GAPotentialTest(unittest.TestCase):
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
        self.potential = GAPotential(name="test")
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
        self.test_struct = d["structure"]

    def test_write_read_cfgs(self):
        self.potential.write_cfgs("test.xyz", cfg_pool=self.test_pool)
        datapool, df = self.potential.read_cfgs("test.xyz")
        self.assertEqual(len(self.test_pool), len(datapool))
        for data1, data2 in zip(self.test_pool, datapool):
            struct1 = data1["structure"]
            struct2 = Structure.from_dict(data2["structure"])
            self.assertTrue(struct1 == struct2)
            energy1 = data1["outputs"]["energy"]
            energy2 = data2["outputs"]["energy"]
            self.assertAlmostEqual(energy1, energy2)
            forces1 = np.array(data1["outputs"]["forces"])
            forces2 = data2["outputs"]["forces"]
            np.testing.assert_array_almost_equal(forces1, forces2)
            stress1 = np.array(data1["outputs"]["virial_stress"])
            stress2 = data2["outputs"]["virial_stress"]
            np.testing.assert_array_almost_equal(stress1, stress2)

    @unittest.skipIf(not which("gap_fit"), "No QUIP cmd found.")
    def test_train(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        self.assertTrue(self.potential.param)

    @unittest.skipIf(not which("quip"), "No QUIP cmd found.")
    def test_evaluate(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        df_orig, df_tar = self.potential.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

    @unittest.skipIf(not which("gap_fit"), "No QUIP cmd found.")
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_predict(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        energy, forces, stress = self.potential.predict_efs(self.test_struct)
        self.assertEqual(len(forces), len(self.test_struct))
        self.assertEqual(len(stress), 6)


if __name__ == "__main__":
    unittest.main()
