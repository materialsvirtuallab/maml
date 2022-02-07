# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import os
import shutil
import tempfile
import unittest

from monty.os.path import which
from monty.serialization import loadfn
from sklearn.linear_model import LinearRegression

from maml.apps.pes import SNAPotential
from maml.base import SKLModel
from maml.describers import BispectrumCoefficients

CWD = os.getcwd()
DIR = os.path.abspath(os.path.dirname(__file__))
test_datapool = loadfn(os.path.join(DIR, "datapool.json"))
coeff_file = os.path.join(DIR, "SNAP", "SNAPotential.snapcoeff")
param_file = os.path.join(DIR, "SNAP", "SNAPotential.snapparam")


@unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
class SNAPotentialTest(unittest.TestCase):
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
        profile = {"Mo": {"r": 0.6, "w": 1.0}}
        self.describer1 = BispectrumCoefficients(
            rcutfac=4.6, twojmax=6, element_profile=profile, quadratic=False, pot_fit=True
        )
        model1 = SKLModel(describer=self.describer1, model=LinearRegression())
        self.potential1 = SNAPotential(model=model1, name="test")
        self.describer2 = BispectrumCoefficients(
            rcutfac=4.6, twojmax=6, element_profile=profile, quadratic=True, pot_fit=True
        )
        model2 = SKLModel(describer=self.describer2, model=LinearRegression())
        self.potential2 = SNAPotential(model=model2, name="test")
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

    def test_train(self):
        self.potential1.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        self.assertEqual(len(self.potential1.model.model.coef_), len(self.describer1.subscripts) + 1)
        self.potential2.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        nss = len(self.describer2.subscripts)
        self.assertEqual(len(self.potential2.model.model.coef_), nss + int((1 + nss) * nss / 2) + 1)

    def test_evaluate(self):
        self.potential1.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        df_orig, df_tar = self.potential1.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

        self.potential2.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        df_orig, df_tar = self.potential2.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

    def test_predict_efs(self):
        self.potential1.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        energy, forces, stress = self.potential1.predict_efs(self.test_struct)
        self.assertEqual(len(forces), len(self.test_struct))
        self.assertEqual(len(stress), 6)
        self.potential2.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        energy, forces, stress = self.potential2.predict_efs(self.test_struct)
        self.assertEqual(len(forces), len(self.test_struct))
        self.assertEqual(len(stress), 6)

    def test_from_config(self):
        snap = SNAPotential.from_config(param_file, coeff_file)
        self.assertTrue(getattr(snap.model.model, "coef_") is not None)


if __name__ == "__main__":
    unittest.main()
