# coding: utf-8

import unittest

from pymatgen.util.testing import PymatgenTest

from maml.describer import ElementProperty


class ElementPropertyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s = PymatgenTest.get_structure("Li2O")

    def test_element_property(self):
        ep = ElementProperty.from_preset("magpie")
        res = ep.transform([self.s.composition])
        self.assertEqual(res.shape, (1, 132))


if __name__ == "__main__":
    unittest.main()
