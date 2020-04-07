# coding: utf-8

import unittest
import os

import numpy as np
from pymatgen.util.testing import PymatgenTest
from pymatgen import Composition, Molecule

from maml.describer import ElementProperty, ElementStats


CWD = os.path.abspath(os.path.dirname(__file__))


class ElementPropertyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s = PymatgenTest.get_structure("Li2O")

    def test_element_property(self):
        ep = ElementProperty.from_preset("magpie")
        res = ep.transform([self.s.composition])
        self.assertEqual(res.shape, (1, 132))


class ElementStatsTest(unittest.TestCase):

    def test_element_stats(self):
        test = ElementStats({'H': [4, 2, 3],
                             'O': [2, 3, 4]},
                            stats=['min', 'max', 'moment:1:10'])
        self.assertEqual(test.transform(['H2O']).shape, (1, 36))

        res = test.transform(['H2O', 'H2O'])
        self.assertEqual(res.shape, (2, 36))

        res2 = test.transform([Composition('H2O'), Composition('H2O')])
        np.testing.assert_allclose(res.values, res2.values)

        dummy_h2o = Molecule(['H', 'H', 'O'], [[-0.5, 0, 0], [0.5, 0, 0], [0, 0, 0]])
        res3 = test.transform([dummy_h2o, dummy_h2o])
        np.testing.assert_allclose(res.values, res3.values)

        stats = ['min', 'max', *['moment:%d:None' % i for i in range(1, 11)]]
        p0_names = ['p0_%s' % i for i in stats]
        p1_names = ['p1_%s' % i for i in stats]
        p2_names = ['p2_%s' % i for i in stats]
        all_names = []
        for i, j, k in zip(p0_names, p1_names, p2_names):
            all_names.extend([i, j, k])

        self.assertListEqual(list(res.columns), all_names)

    def test_from_data(self):
        es = ElementStats.from_data('megnet_1', stats=['moment:None:5'])
        d = es.transform_one('Fe2O3')
        self.assertTrue(d.shape == (1, 80))

        es = ElementStats.from_data(['megnet_1'], stats=['moment:None:5'])
        d = es.transform_one('Fe2O3')
        self.assertTrue(d.shape == (1, 80))

        es2 = ElementStats.from_data(['megnet_1', 'megnet_3'], stats=['moment:None:5'])
        d = es2.transform_one('Fe2O3')
        self.assertTrue(d.shape == (1, 160))

    def test_error(self):
        with self.assertRaises(ValueError):
            _ = ElementStats(element_properties={"H": [1, 2], "O": [1, 2, 3]}, stats=['mean'])

        with self.assertRaises(ValueError):
            _ = ElementStats(element_properties={"H": [1, 2], "O": [1, 2]}, stats=['mean'],
                             property_names=['p1'])

        with self.assertRaises(ValueError):
            _ = ElementStats(element_properties={"H": [1, 2], "O": [1, 2]}, stats=['super_std'],
                             property_names=['p1', 'p2'])

        with self.assertRaises(ValueError):
            _ = ElementStats(element_properties={"H": [1, 2], "O": [1, 2]}, stats=['super_std:-1'],
                             property_names=['p1', 'p2'])

        with self.assertRaises(ValueError):
            ElementStats.from_file(os.path.join(CWD, 'test_data/wrong_dummy_property.json'), stats=['max'])

    def test_keys(self):
        es = ElementStats.from_file(os.path.join(CWD, 'test_data/dummy_property.json'), stats=['max'])
        self.assertListEqual(es.stats, ['mean'])


if __name__ == "__main__":
    unittest.main()
