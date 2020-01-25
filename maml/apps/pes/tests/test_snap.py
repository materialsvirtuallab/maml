# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import unittest
import tempfile
import os
import shutil

from monty.os.path import which
from monty.serialization import loadfn
from maml.apps.pes.snap import SNAPotential
from maml.model.linear_model import LinearModel
from maml.describer.atomic_describer import BispectrumCoefficients

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.dirname(__file__), 'datapool.json'))


@unittest.skipIf(not which('lmp_serial'), 'No LAMMPS cmd found.')
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
        profile = {'Mo': {'r': 0.6, 'w': 1.}}
        self.describer1 = BispectrumCoefficients(rcutfac=4.6, twojmax=6,
                                                 element_profile=profile,
                                                 quadratic=False,
                                                 pot_fit=True)
        model1 = LinearModel(describer=self.describer1)
        self.potential1 = SNAPotential(model=model1, name='test')
        self.describer2 = BispectrumCoefficients(rcutfac=4.6, twojmax=6,
                                                 element_profile=profile,
                                                 quadratic=True,
                                                 pot_fit=True)
        model2 = LinearModel(describer=self.describer2)
        self.potential2 = SNAPotential(model=model2, name='test')
        self.test_pool = test_datapool
        self.test_structures = []
        self.test_energies = []
        self.test_forces = []
        self.test_stresses = []
        for d in self.test_pool:
            self.test_structures.append(d['structure'])
            self.test_energies.append(d['outputs']['energy'])
            self.test_forces.append(d['outputs']['forces'])
            self.test_stresses.append(d['outputs']['virial_stress'])
        self.test_struct = d['structure']

    def test_train(self):
        self.potential1.train(train_structures=self.test_structures,
                              energies=self.test_energies,
                              forces=self.test_forces,
                              stresses=self.test_stresses)
        self.assertEqual(len(self.potential1.model.coef),
                         len(self.describer1.subscripts) + 1)
        self.potential2.train(train_structures=self.test_structures,
                              energies=self.test_energies,
                              forces=self.test_forces,
                              stresses=self.test_stresses)
        nss = len(self.describer2.subscripts)
        self.assertEqual(len(self.potential2.model.coef),
                         nss + int((1 + nss) * nss / 2) + 1)

    def test_evaluate(self):
        self.potential1.train(train_structures=self.test_structures,
                              energies=self.test_energies,
                              forces=self.test_forces,
                              stresses=self.test_stresses)
        df_orig, df_tar = self.potential1.evaluate(test_structures=self.test_structures,
                                                   ref_energies=self.test_energies,
                                                   ref_forces=self.test_forces,
                                                   ref_stresses=self.test_stresses)
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

        self.potential2.train(train_structures=self.test_structures,
                              energies=self.test_energies,
                              forces=self.test_forces,
                              stresses=self.test_stresses)
        df_orig, df_tar = self.potential2.evaluate(test_structures=self.test_structures,
                                                   ref_energies=self.test_energies,
                                                   ref_forces=self.test_forces,
                                                   ref_stresses=self.test_stresses)
        self.assertEqual(df_orig.shape[0], df_tar.shape[0])

    def test_predict(self):
        self.potential1.train(train_structures=self.test_structures,
                              energies=self.test_energies,
                              forces=self.test_forces,
                              stresses=self.test_stresses)
        energy, forces, stress = self.potential1.predict(self.test_struct)
        self.assertEqual(len(forces), len(self.test_struct))
        self.assertEqual(len(stress), 6)
        self.potential2.train(train_structures=self.test_structures,
                              energies=self.test_energies,
                              forces=self.test_forces,
                              stresses=self.test_stresses)
        energy, forces, stress = self.potential2.predict(self.test_struct)
        self.assertEqual(len(forces), len(self.test_struct))
        self.assertEqual(len(stress), 6)


if __name__ == '__main__':
    unittest.main()
