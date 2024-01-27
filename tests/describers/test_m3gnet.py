from __future__ import annotations

import numpy as np
import pytest
import unittest

from pymatgen.core import Lattice, Structure

try:
    from m3gnet.models import M3GNet
except ImportError:
    M3GNet = None

from maml.describers import M3GNetStructure, M3GNetSite


@unittest.skipIf(M3GNet is None, "M3GNet package is required.")
class M3GNetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]]
        )

    def test_m3gnet_site_transform(self):
        atom_feat_2s = M3GNetSite(feature_batch="pandas_concat").transform([self.s] * 2)
        self.assertListEqual(list(np.array(atom_feat_2s).shape), [16, 64])
        with pytest.raises(ValueError, match="Invalid output_layers"):
            M3GNetSite(output_layers=["whatever"])
        atom_feat_2s_2l = M3GNetSite(
            output_layers=["embedding", "gc_3"], feature_batch="pandas_concat"
        ).transform([self.s] * 2)
        self.assertListEqual(list(np.array(atom_feat_2s_2l).shape), [16, 128])
        atom_feat_dict = M3GNetSite(return_type=dict).transform_one(self.s)
        assert type(atom_feat_dict) == dict

    def test_m3gnet_structure_transform(self):
        struct_feat_2s = M3GNetStructure().transform([self.s] * 2)
        self.assertListEqual(list(np.array(struct_feat_2s).shape), [2, 128])


if __name__ == "__main__":
    unittest.main()
