# coding: utf-8

import unittest

from pymatgen.util.testing import PymatgenTest

from maml.describer import ElementProperty, ElementStats


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

        stats = ['min', 'max', *['moment:%d:None' % i for i in range(1, 11)]]
        p0_names = ['p0_%s' % i for i in stats]
        p1_names = ['p1_%s' % i for i in stats]
        p2_names = ['p2_%s' % i for i in stats]
        all_names = []
        for i, j, k in zip(p0_names, p1_names, p2_names):
            all_names.extend([i, j, k])

        self.assertListEqual(list(res.columns), all_names)


if __name__ == "__main__":
    unittest.main()
