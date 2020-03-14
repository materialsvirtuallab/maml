# coding: utf-8

import os
import tempfile
import unittest
import itertools

import numpy as np
from monty.os.path import which
from pymatgen import Lattice, Structure

from maml.describer._site import \
    BispectrumCoefficients, SmoothOverlapAtomicPosition, BPSymmetryFunctions


class BispectrumCoefficientsTest(unittest.TestCase):

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found")
    def test_transform(self):
        s = Structure.from_spacegroup('Fm-3m', Lattice.cubic(5.69169),
                                      ['Na', 'Cl'], [[0, 0, 0], [0, 0, 0.5]])
        profile = dict(Na=dict(r=0.3, w=0.9),
                       Cl=dict(r=0.7, w=3.0))
        s *= [2, 2, 2]
        structures = [s] * 10
        for s in structures:
            n = np.random.randint(4)
            inds = np.random.randint(16, size=n)
            s.remove_sites(inds)

        bc_atom = BispectrumCoefficients(cutoff=5, twojmax=3, element_profile=profile,
                                         quadratic=False, pot_fit=False)
        df_atom = bc_atom.transform(structures)
        for i, s in enumerate(structures):
            df_s = df_atom.xs(i, level='input_index')
            self.assertEqual(df_s.shape, (len(s), 8))
            self.assertTrue(df_s.equals(bc_atom.transform_one(s)))

        bc_pot = BispectrumCoefficients(cutoff=5, twojmax=3, element_profile=profile,
                                        quadratic=False, pot_fit=True, include_stress=True)
        df_pot = bc_pot.transform(structures)
        for i, s in enumerate(structures):
            df_s = df_pot.xs(i, level='input_index')
            self.assertEqual(df_s.shape, (1 + len(s) * 3 + 6, 18))
            self.assertTrue(df_s.equals(bc_pot.transform_one(s)))
            sna = df_s.iloc[0]
            for specie in ['Na', 'Cl']:
                self.assertAlmostEqual(
                    sna[specie, 'n'],
                    s.composition.fractional_composition[specie])
                np.testing.assert_array_equal(df_s[specie, 'n'][1:],
                                              np.zeros(len(s) * 3 + 6))


class SmoothOverlapAtomicPositionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup("Im-3m", Lattice.cubic(3.4268),
                                                      ["Li"], [[0, 0, 0]])
        self.binary_struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169),
                                                       ['Na', 'Cl'], [[0, 0, 0], [0, 0, 0.5]])
        self.describer = SmoothOverlapAtomicPosition(cutoff=4.8, l_max=8, n_max=8)

    @unittest.skipIf(not which('quip'), 'No quip cmd found.')
    def test_transform_one(self):
        unary_descriptors = self.describer.transform_one(self.unary_struct)
        binary_descriptors = self.describer.transform_one(self.binary_struct)
        self.assertEqual(unary_descriptors.shape[0], len(self.unary_struct))
        self.assertEqual(binary_descriptors.shape[0], len(self.binary_struct))

    @unittest.skipIf(not which('quip'), 'No quip cmd found.')
    def test_transform(self):
        unary_descriptors = self.describer.transform([self.unary_struct] * 3)
        self.assertEqual(unary_descriptors.shape[0], len(self.unary_struct) * 3)
        binary_descriptors = self.describer.transform([self.binary_struct] * 3)
        self.assertEqual(binary_descriptors.shape[0], len(self.binary_struct) * 3)


class BPSymmetryFunctionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup("Im-3m", Lattice.cubic(3.4268),
                                                      ["Li"], [[0, 0, 0]])
        self.binary_struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169),
                                                       ['Na', 'Cl'], [[0, 0, 0], [0, 0, 0.5]])
        self.r_etas = [0.01, 0.02, 0.05]
        self.r_shift = [0]
        self.zetas = [1.0, 4.0]
        self.a_etas = [0.01, 0.05]
        self.lambdas = [-1, 1]
        self.describer = BPSymmetryFunctions(cutoff=4.8, r_etas=self.r_etas, r_shift=self.r_shift, 
                                             lambdas=self.lambdas, a_etas=self.a_etas, zetas=self.zetas)

    def test_transform_one(self):
        unary_descriptors = self.describer.transform_one(self.unary_struct)
        binary_descriptors = self.describer.transform_one(self.binary_struct)
        unary_elements = set(self.unary_struct.species)
        binary_elements = set(self.binary_struct.species)
        self.assertEqual(unary_descriptors.shape[0], len(self.unary_struct))
        self.assertEqual(unary_descriptors.shape[1],
                         len(self.r_etas) * len(self.r_shift) * len(unary_elements) +
                         len(self.a_etas) * len(self.zetas) * len(self.lambdas)
                         * len(list(itertools.combinations_with_replacement(unary_elements, 2))))
        self.assertEqual(binary_descriptors.shape[0], len(self.binary_struct))
        self.assertEqual(binary_descriptors.shape[1],
                         len(self.r_etas) * len(self.r_shift) * len(binary_elements) +
                         len(self.a_etas) * len(self.zetas) * len(self.lambdas)
                         * len(list(itertools.combinations_with_replacement(binary_elements, 2))))

    def test_transform(self):
        unary_descriptors = self.describer.transform([self.unary_struct] * 3)
        binary_descriptors = self.describer.transform([self.binary_struct] * 3)
        unary_elements = set(self.unary_struct.species)
        binary_elements = set(self.binary_struct.species)
        self.assertEqual(unary_descriptors.shape[0], len(self.unary_struct) * 3)
        self.assertEqual(unary_descriptors.shape[1],
                         len(self.r_etas) * len(self.r_shift) * len(unary_elements) +
                         len(self.a_etas) * len(self.zetas) * len(self.lambdas)
                         * len(list(itertools.combinations_with_replacement(unary_elements, 2))))
        self.assertEqual(binary_descriptors.shape[0], len(self.binary_struct) * 3)
        self.assertEqual(binary_descriptors.shape[1],
                         len(self.r_etas) * len(self.r_shift) * len(binary_elements) +
                         len(self.a_etas) * len(self.zetas) * len(self.lambdas)
                         * len(list(itertools.combinations_with_replacement(binary_elements, 2))))


if __name__ == "__main__":
    unittest.main()
