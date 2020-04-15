# coding: utf-8

import unittest

from pymatgen import Lattice, Structure

from maml.describer import MEGNetSite, MEGNetStructure


class MEGNETTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s = Structure.from_spacegroup(
            'Fm-3m', Lattice.cubic(5.69169),
            ['Na', 'Cl'], [[0, 0, 0], [0, 0, 0.5]])

    def test_megnet_site_transform(self):
        msite = MEGNetSite()
        features = msite.transform_one(self.s)
        self.assertListEqual(list(features.shape), [8, 32])

        msite = MEGNetSite(feature_batch='pandas_concat')
        features2 = msite.transform([self.s, self.s])
        self.assertListEqual(list(features2.shape), [16, 32])

    def test_megnet_structure_transform(self):
        mstruct = MEGNetStructure(mode='site_stats')
        self.assertListEqual(list(mstruct.transform_one(self.s).shape), [1, 32 * 6])

        mstruct = MEGNetStructure(mode='site_readout')
        self.assertListEqual(list(mstruct.transform_one(self.s).shape), [1, 32])

        mstruct = MEGNetStructure(mode='final')
        self.assertListEqual(list(mstruct.transform_one(self.s).shape), [1, 32 * 3])


if __name__ == "__main__":
    unittest.main()
