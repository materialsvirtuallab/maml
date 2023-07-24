from __future__ import annotations

import unittest

from pymatgen.core import Lattice, Structure

try:
    from megnet.models import MEGNetModel
except ImportError:
    MEGNetModel = None

from maml.describers import MEGNetSite, MEGNetStructure


@unittest.skipIf(MEGNetModel is None, "MEGNet package is required")
class MEGNETTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])
        cls.dummy_model = MEGNetModel(100, 2, nblocks=1, n1=4, n2=2, n3=2, npass=1)

    def test_megnet_site_transform(self):
        msite = MEGNetSite(name=self.dummy_model, level=1, feature_batch="pandas_concat")
        features2 = msite.transform([self.s, self.s])
        self.assertListEqual(list(features2.shape), [16, 2])

    def test_megnet_structure_transform(self):
        mstruct = MEGNetStructure(name=self.dummy_model, mode="site_stats", level=1)
        self.assertListEqual(list(mstruct.transform_one(self.s).shape), [1, 2 * 6])

        mstruct.mode = "site_readout"
        self.assertListEqual(list(mstruct.transform_one(self.s).shape), [1, 4])

        mstruct.mode = "final"
        self.assertListEqual(list(mstruct.transform_one(self.s).shape), [1, 10])


if __name__ == "__main__":
    unittest.main()
