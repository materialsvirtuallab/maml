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

from maml.apps.pes import MTPotential

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.abspath(os.path.dirname(__file__)), "datapool.json"))
config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "MTP", "fitted.mtp")


class MTPotentialTest(unittest.TestCase):
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
        self.potential = MTPotential(name="test")
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
        self.test_struct = self.test_pool[-1]["structure"]

    def test_write_read_cfgs(self):
        self.potential.elements = ["Mo"]
        self.potential.write_cfg("test.cfgs", cfg_pool=self.test_pool)
        datapool, df = self.potential.read_cfgs("test.cfgs")
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

    @unittest.skipIf(not which("mlp"), "No MLIP cmd found.")
    def test_train(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            unfitted_mtp="08g.mtp",
            max_dist=3.0,
            max_iter=20,
        )
        self.assertTrue(self.potential.param)

    @unittest.skipIf(not which("mlp"), "No MLIP cmd found.")
    def test_evaluate(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            unfitted_mtp="08g.mtp",
            max_dist=3.0,
            max_iter=20,
        )
        df_orig, df_tar = self.potential.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

    @unittest.skipIf(not which("mlp"), "No MLIP cmd found.")
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_predict_efs(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            unfitted_mtp="08g.mtp",
            max_dist=3.0,
            max_iter=20,
        )
        energy, forces, stress = self.potential.predict_efs(self.test_struct)
        self.assertEqual(len(forces), len(self.test_struct))
        self.assertEqual(len(stress), 6)

    def test_from_config(self):
        mtp = MTPotential.from_config(config_file, elements=["Mo"])
        self.assertIsNotNone(mtp.param)


if __name__ == "__main__":
    unittest.main()
