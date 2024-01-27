from __future__ import annotations

import numpy as np
import unittest

from pymatgen.core import Lattice, Structure

try:
    from m3gnet.models import M3GNet
except ImportError:
    M3GNet = None

from maml.describers import M3GNetStructure, M3GNetSite


@unittest.skipIf(M3GNet is None, "M3GNet package is required")
class M3GNetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]]
        )
        cls.m3gnet_struct = M3GNetStructure()
        cls.m3gnet_site = M3GNetSite(feature_batch="pandas_concat")

    def test_m3gnet_site_transform(self):
        atom_features2 = self.m3gnet_site.transform([self.s] * 2)
        self.assertListEqual(list(np.array(atom_features2).shape), [16, 64])

    def test_m3gnet_structure_transform(self):
        struct_feature2 = self.m3gnet_struct.transform([self.s] * 2)
        self.assertListEqual(list(np.array(struct_feature2).shape), [2, 128])


if __name__ == "__main__":
    unittest.main()
