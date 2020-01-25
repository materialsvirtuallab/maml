# coding: utf-8

import unittest
import os
import pandas as pd
import numpy as np
from pymatgen import Structure, Element, Lattice
from pymatgen.util.testing import PymatgenTest
from maml.describer.structural_describer import DistinctSiteProperty, CoulombMatrix


class DistinctSitePropertyTest(PymatgenTest):

    @classmethod
    def setUpClass(cls):
        cls.li2o = cls.get_structure("Li2O")
        cls.na2o = cls.li2o.copy()
        cls.na2o["Li+"] = "Na+"
        cls.describer = DistinctSiteProperty(['2c', '1a'],
                                             ["Z", "atomic_radius"])

    def test_describe(self):
        descriptor = self.describer.describe(self.li2o)
        self.assertAlmostEqual(descriptor.iloc[0]["2c-Z"], 3)
        self.assertAlmostEqual(descriptor.iloc[0]["2c-atomic_radius"], 1.45)
        descriptor = self.describer.describe(self.na2o)
        self.assertEqual(descriptor.iloc[0]["1a-Z"], 8)
        self.assertEqual(descriptor.iloc[0]["1a-atomic_radius"], 0.6)

    def test_describe_all(self):
        df = pd.DataFrame(self.describer.describe_all([self.li2o, self.na2o]))
        print(df)
        self.assertEqual(df.iloc[0]["2c-Z"], 3)
        self.assertEqual(df.iloc[0]["2c-atomic_radius"], 1.45)


class CoulomMatrixTest(unittest.TestCase):

    def setUp(self):

        self.s1 = Structure.from_spacegroup(225,
                                            Lattice.cubic(5.69169),
                                            ["Na", "Cl"],
                                            [[0, 0, 0], [0, 0, 0.5]])
        self.s2 = Structure.from_dict({'@class': 'Structure',
                                       '@module': 'pymatgen.core.structure',
                                       'charge': None,
                                       'lattice': {'a': 5.488739045730133,
                                                   'alpha': 60.0000000484055,
                                                   'b': 5.488739048031658,
                                                   'beta': 60.00000003453459,
                                                   'c': 5.48873905,
                                                   'gamma': 60.000000071689925,
                                                   'matrix': [[4.75338745, 0.0, 2.74436952],
                                                              [1.58446248, 4.48153667, 2.74436952],
                                                              [0.0, 0.0, 5.48873905]],
                                                   'volume': 116.92375473740876},
                                       'sites': [{'abc': [0.5, 0.5, 0.5],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [3.168924965, 2.240768335, 5.488739045]},
                                                 {'abc': [0.5, 0.5, 0.0],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [3.168924965, 2.240768335, 2.74436952]},
                                                 {'abc': [0.0, 0.5, 0.5],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [0.79223124, 2.240768335, 4.116554285]},
                                                 {'abc': [0.5, 0.0, 0.5],
                                                  'label': 'Al',
                                                  'properties': {'coordination_no': 10, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Al', 'occu': 1}],
                                                  'xyz': [2.376693725, 0.0, 4.116554285]},
                                                 {'abc': [0.875, 0.875, 0.875],
                                                  'label': 'Lu',
                                                  'properties': {'coordination_no': 16, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Lu', 'occu': 1}],
                                                  'xyz': [5.54561868875, 3.9213445862499996, 9.60529332875]},
                                                 {'abc': [0.125, 0.125, 0.125],
                                                  'label': 'Lu',
                                                  'properties': {'coordination_no': 16, 'forces': [0.0, 0.0, 0.0]},
                                                  'species': [{'element': 'Lu', 'occu': 1}],
                                                  'xyz': [0.79223124125, 0.56019208375, 1.37218476125]}]})

    def test_coulomb_mat(self):
        cm = CoulombMatrix()
        cmat = cm.describe(self.s1).values.reshape(self.s1.num_sites, self.s1.num_sites)
        na = Element('Na')
        cl = Element('Cl')
        dist = self.s1.distance_matrix
        self.assertEqual(cmat[0][0], (na.Z ** 2.4) * 0.5)
        self.assertEqual(cmat[4][4], (cl.Z ** 2.4) * 0.5)
        self.assertEqual(cmat[0][1], (na.Z * na.Z) / dist[0][1])

    def test_sorted_coulomb_mat(self):
        cm = CoulombMatrix(sorted=True)
        c = cm.coulomb_mat(self.s2)
        cmat = cm.describe(self.s2).values.reshape(self.s2.num_sites, self.s2.num_sites)
        norm_order_ind = np.argsort(np.linalg.norm(c, axis=1))
        for i in range(cmat.shape[1]):
            self.assertTrue(np.all(cmat[i] == c[norm_order_ind[i]]))

    def test_random_coulom_mat(self):
        cm = CoulombMatrix(randomized=True, random_seed=7)
        c = cm.coulomb_mat(self.s2)
        cmat = cm.describe(self.s2).values.reshape(self.s2.num_sites, self.s2.num_sites)
        cm2 = CoulombMatrix(randomized=True, random_seed=8)
        cmat2 = cm2.describe(self.s2).values.reshape(self.s2.num_sites, self.s2.num_sites)
        self.assertEqual(np.all(cmat == cmat2), False)
        for i in range(cmat.shape[1]):
            self.assertTrue(cmat[i] in c[i])

    def test_describe_all(self):
        cm = CoulombMatrix()
        c = cm.describe_all([self.s1, self.s2])
        c1 = cm.describe(self.s1)
        c2 = cm.describe(self.s2)
        self.assertTrue(np.all(c[0].dropna() == c1[0]))
        self.assertTrue(np.all(c[1].dropna() == c2[0]))


if __name__ == "__main__":
    unittest.main()
