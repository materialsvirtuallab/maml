# coding: utf-8

import unittest
from pymatgen.util.testing import PymatgenTest

import os
import tempfile
import numpy as np
from monty.os.path import which
from pymatgen import Lattice, Structure, Element

from maml.describer.atomic_describer import \
    BispectrumCoefficients, SOAPDescriptor, BPSymmetryFunctions


class BispectrumCoefficientsTest(unittest.TestCase):

    @staticmethod
    def test_subscripts():

        def from_lmp_doc(twojmax, diagonal):
            js = []
            for j1 in range(0, twojmax + 1):
                if diagonal == 2:
                    js.append([j1, j1, j1])
                elif diagonal == 1:
                    for j in range(0, min(twojmax, 2 * j1) + 1, 2):
                        js.append([j1, j1, j])
                elif diagonal == 0:
                    for j2 in range(0, j1 + 1):
                        for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                            js.append([j1, j2, j])
                elif diagonal == 3:
                    for j2 in range(0, j1 + 1):
                        for j in range(j1 - j2, min(twojmax, j1 + j2) + 1, 2):
                            if j >= j1:
                                js.append([j1, j2, j])
            return js

        profile = {"Mo": {"r": 0.5, "w": 1}}
        for d in range(4):
            for tjm in range(11):
                bc = BispectrumCoefficients(1.0, twojmax=tjm,
                                            element_profile=profile,
                                            quadratic=False,
                                            diagonalstyle=d)
                np.testing.assert_equal(bc.subscripts, from_lmp_doc(tjm, d))

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found")
    def test_describe(self):
        s = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                      ['Na', 'Cl'],
                                      [[0, 0, 0], [0, 0, 0.5]])
        profile = dict(Na=dict(r=0.3, w=0.9),
                       Cl=dict(r=0.7, w=3.0))
        s *= [2, 2, 2]
        structures = [s] * 10
        for s in structures:
            n = np.random.randint(4)
            inds = np.random.randint(16, size=n)
            s.remove_sites(inds)

        bc_atom = BispectrumCoefficients(5, 3, profile, diagonalstyle=2,
                                         quadratic=False, pot_fit=False)
        df_atom = bc_atom.describe_all(structures)
        for i, s in enumerate(structures):
            df_s = df_atom.xs(i, level='input_index')
            self.assertEqual(df_s.shape, (len(s), 4))
            self.assertTrue(df_s.equals(bc_atom.describe(s)))

        bc_pot = BispectrumCoefficients(5, 3, profile, diagonalstyle=2,
                                        quadratic=False, pot_fit=True)
        df_pot = bc_pot.describe_all(structures, include_stress=True)
        for i, s in enumerate(structures):
            df_s = df_pot.xs(i, level='input_index')
            self.assertEqual(df_s.shape, ((1 + len(s) * 3 + 6, 10)))
            self.assertTrue(df_s.equals(bc_pot.describe(s, include_stress=True)))
            sna = df_s.iloc[0]
            for specie in ['Na', 'Cl']:
                self.assertAlmostEqual(
                    sna[specie, 'n'],
                    s.composition.fractional_composition[specie])
                np.testing.assert_array_equal(df_s[specie, 'n'][1:],
                                              np.zeros(len(s) * 3 + 6))


class SOAPDescriptorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup('Im-3m', Lattice.cubic(3.4268),
                                [{"Li": 1}], [[0, 0, 0]])
        self.binary_struct = Structure.from_spacegroup(225, Lattice.cubic(5.69169),
                                                       ['Na', 'Cl'],
                                                       [[0, 0, 0], [0, 0, 0.5]])
        self.describer = SOAPDescriptor(cutoff=4.8, l_max=8, n_max=8)

    @unittest.skipIf(not which('quip'), 'No quip cmd found.')
    def test_describe(self):
        unary_descriptors = self.describer.describe(self.unary_struct)
        binary_descriptors = self.describer.describe(self.binary_struct)
        self.assertEqual(unary_descriptors.shape[0], len(self.unary_struct))
        self.assertEqual(binary_descriptors.shape[0], len(self.binary_struct))

    @unittest.skipIf(not which('quip'), 'No quip cmd found.')
    def test_describe_all(self):
        unary_descriptors = self.describer.describe_all([self.unary_struct] * 3)
        self.assertEqual(unary_descriptors.shape[0], len(self.unary_struct) * 3)
        binary_descriptors = self.describer.describe_all([self.binary_struct] * 3)
        self.assertEqual(binary_descriptors.shape[0], len(self.binary_struct) * 3)


class BPSymmetryFunctionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup('Im-3m', Lattice.cubic(3.4268),
                                [{"Li": 1}], [[0, 0, 0]])
        self.num_symm2 = 3
        self.a_etas = [0.01, 0.05]
        self.describer = BPSymmetryFunctions(dmin=2.0, cutoff=4.8,
                                             num_symm2=self.num_symm2,
                                             a_etas=self.a_etas)

    @unittest.skipIf(not which('RuNNerMakesym'), 'No RuNNerMakesym cmd found.')
    @unittest.skipIf(not which('RuNNer'), 'No RuNNer cmd found.')
    def test_describe(self):
        unary_descriptors = self.describer.describe(self.unary_struct)
        self.assertEqual(unary_descriptors.shape[0], len(self.unary_struct))
        self.assertEqual(unary_descriptors.shape[1],
                         self.num_symm2 + len(self.a_etas) * 2 * 4)

    @unittest.skipIf(not which('RuNNerMakesym'), 'No RuNNerMakesym cmd found.')
    @unittest.skipIf(not which('RuNNer'), 'No RuNNer cmd found.')
    def test_describe_all(self):
        descriptors = self.describer.describe([self.unary_struct] * 3)
        self.assertEqual(descriptors.shape[0], len(self.unary_struct) * 3)
        self.assertEqual(descriptors.shape[1],
                         self.num_symm2 + len(self.a_etas) * 2 * 4)

if __name__ == "__main__":
    unittest.main()