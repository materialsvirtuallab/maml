from __future__ import annotations

import itertools
import os
import tempfile
import unittest

import numpy as np
from shutil import which
from pymatgen.core import Lattice, Structure

from maml.describers._site import (
    BispectrumCoefficients,
    BPSymmetryFunctions,
    SiteElementProperty,
    SmoothOverlapAtomicPosition,
)


class BispectrumCoefficientsTest(unittest.TestCase):
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found")
    def test_transform(self):
        s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])
        profile = dict(Na=dict(r=0.3, w=0.9), Cl=dict(r=0.7, w=3.0))
        s *= [2, 2, 2]
        structures = [s] * 10
        for s in structures:
            n = np.random.randint(4)
            inds = np.random.randint(16, size=n)
            s.remove_sites(inds)

        bc_atom = BispectrumCoefficients(rcutfac=5, twojmax=3, element_profile=profile, quadratic=False, pot_fit=False)
        df_atom = bc_atom.transform(structures)
        for i, s in enumerate(structures):
            df_s = df_atom.xs(i, level="input_index")
            assert df_s.shape == (len(s), 8)
            assert df_s.equals(bc_atom.transform_one(s))

        bc_pot = BispectrumCoefficients(
            rcutfac=5, twojmax=3, element_profile=profile, quadratic=False, pot_fit=True, include_stress=True
        )
        df_pot = bc_pot.transform(structures)
        for i, s in enumerate(structures):
            df_s = df_pot.xs(i, level="input_index")
            assert df_s.shape == (1 + len(s) * 3 + 6, 18)
            assert df_s.equals(bc_pot.transform_one(s))
            sna = df_s.iloc[0]
            for specie in ["Na", "Cl"]:
                self.assertAlmostEqual(sna[specie, "n"], s.composition.fractional_composition[specie])
                np.testing.assert_array_equal(df_s[specie, "n"][1:], np.zeros(len(s) * 3 + 6))


class SmoothOverlapAtomicPositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup("Im-3m", Lattice.cubic(3.4268), ["Li"], [[0, 0, 0]])
        self.binary_struct = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]]
        )
        self.describer = SmoothOverlapAtomicPosition(cutoff=4.8, l_max=8, n_max=8)

    @unittest.skipIf(not which("quip"), "No quip cmd found.")
    def test_transform_one(self):
        unary_descriptors = self.describer.transform_one(self.unary_struct)
        binary_descriptors = self.describer.transform_one(self.binary_struct)
        assert unary_descriptors.shape[0] == len(self.unary_struct)
        assert binary_descriptors.shape[0] == len(self.binary_struct)

    @unittest.skipIf(not which("quip"), "No quip cmd found.")
    def test_transform(self):
        unary_descriptors = self.describer.transform([self.unary_struct] * 3)
        assert unary_descriptors.shape[0] == len(self.unary_struct) * 3
        binary_descriptors = self.describer.transform([self.binary_struct] * 3)
        assert binary_descriptors.shape[0] == len(self.binary_struct) * 3


class BPSymmetryFunctionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup("Im-3m", Lattice.cubic(3.4268), ["Li"], [[0, 0, 0]])
        self.binary_struct = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]]
        )
        self.r_etas = [0.01, 0.02, 0.05]
        self.r_shift = [0]
        self.zetas = [1.0, 4.0]
        self.a_etas = [0.01, 0.05]
        self.lambdas = [-1, 1]
        self.describer = BPSymmetryFunctions(
            cutoff=4.8,
            r_etas=self.r_etas,
            r_shift=self.r_shift,
            lambdas=self.lambdas,
            a_etas=self.a_etas,
            zetas=self.zetas,
        )

    def test_transform_one(self):
        unary_descriptors = self.describer.transform_one(self.unary_struct)
        binary_descriptors = self.describer.transform_one(self.binary_struct)
        unary_elements = set(self.unary_struct.species)
        binary_elements = set(self.binary_struct.species)
        assert unary_descriptors.shape[0] == len(self.unary_struct)
        assert unary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(unary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(unary_elements, 2)))
        assert binary_descriptors.shape[0] == len(self.binary_struct)
        assert binary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(binary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(binary_elements, 2)))

    def test_transform(self):
        unary_descriptors = self.describer.transform([self.unary_struct] * 3)
        binary_descriptors = self.describer.transform([self.binary_struct] * 3)
        unary_elements = set(self.unary_struct.species)
        binary_elements = set(self.binary_struct.species)
        assert unary_descriptors.shape[0] == len(self.unary_struct) * 3
        assert unary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(unary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(unary_elements, 2)))
        assert binary_descriptors.shape[0] == len(self.binary_struct) * 3
        assert binary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(binary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(binary_elements, 2)))


class TestSiteSpecieProperty(unittest.TestCase):
    def test_unordered_site(self):
        s = Structure(Lattice.cubic(3), ["Mo", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        udescriber = SiteElementProperty()
        np.testing.assert_array_almost_equal(udescriber.transform_one(s), np.array([[42, 16]]).T)

        udescriber2 = SiteElementProperty(feature_dict={16: [16, 16], 42: [42, 42]})
        np.testing.assert_array_almost_equal(udescriber2.transform_one(s), np.array([[42, 42], [16, 16]]))

        assert udescriber.describer_type == "site"

        udescriber = SiteElementProperty(output_weights=True)
        vec, weight = udescriber.transform_one(s)
        np.testing.assert_array_almost_equal(vec, np.array([[42, 16]]).T)
        np.testing.assert_array_almost_equal(weight, np.array([1, 1]))

        s2 = s.copy()
        s2.replace_species({"Mo": {"S": 0.1, "Mo": 0.9}})
        self.assertRaises(ValueError, udescriber2.transform_one, s2)
        vec, weight = udescriber.transform_one(s2)
        np.testing.assert_array_almost_equal(vec, np.array([[16, 42, 16]]).T)
        np.testing.assert_array_almost_equal(weight, np.array([0.1, 0.9, 1]))


if __name__ == "__main__":
    unittest.main()
