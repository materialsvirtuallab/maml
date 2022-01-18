import os
import unittest
import numpy as np
from pymatgen.core import Structure
from pymatgen.util.testing import PymatgenTest
import maml.apps.gbe as gbe
from maml.apps.gbe.utils import load_data, load_b0_dict
from maml.apps.gbe.describer import (GBDescriber, GBBond,
                                     get_structural_feature,
                                     get_elemental_feature,
                                     convert_hcp_direction,
                                     convert_hcp_plane)
from maml.apps.gbe.presetfeatures import (e_coh, G, a0, ar,
                                          mean_delta_bl, hb, CLTE,
                                          bdensity, d_gb, d_rot,
                                          sin_theta, cos_theta, e_gb)

pjoin = os.path.join
module_dir = os.path.dirname(gbe.__file__)
REFS = pjoin(module_dir, "references")

class TestDescriber(PymatgenTest):
    def setUp(self) -> None:
        self.db_entries = [d for d in load_data() if d['material_id'] == "mp-91"]  # Contains 10 data
        self.b0_dict = load_b0_dict()

    def test_convert_hcp(self):
        self.assertArrayEqual(convert_hcp_plane((2, -1, -1, 0)), (2, -1, 0))
        self.assertArrayEqual(convert_hcp_direction((2, -1, -1, 0), 'hexagonal'), (1, 0, 0))
        self.assertArrayEqual(convert_hcp_direction((-1, -1, 2, 0), 'hexagonal'), (-1, -1, 0))

    def test_elemental_feature(self):
        # test case: W, theta 109.47122063449069 degree,
        df = get_elemental_feature(self.db_entries[0])
        ans_dict = {e_coh: 8.301059,
                    G: 147,
                    a0: 2.760381,
                    ar: 1.35,
                    mean_delta_bl: -0.004265,
                    hb: 2570.0,
                    CLTE: 4.5 * 1e-6,
                    bdensity: 18.854007561757314}
        for k, v in ans_dict.items():
            self.assertAlmostEqual(df[k.str_name][0], v, places=6)

    def test_structural_feature(self):
        df = get_structural_feature(self.db_entries[0])
        ans_dict = {d_gb: 1.0079488085907933,
                    d_rot: 1.3012563165005835,
                    sin_theta: 0.9428090415820635,
                    cos_theta: -0.33333333333333315}
        for k, v in ans_dict.items():
            self.assertAlmostEqual(df[k.str_name][0], v, places=6)

    def test_describer(self):
        describer = GBDescriber()
        df = describer.transform_one(self.db_entries[0],
                                     inc_target=True,
                                     inc_bulk_ref=True)
        df_gb = df.iloc[0].to_frame().transpose()
        df_bulk = df.iloc[1].to_frame().transpose()
        ans_dict = {e_coh: 8.301059,
                    G: 147,
                    ar: 1.35,
                    mean_delta_bl: -0.004265,
                    d_gb: 1.0079488085907933,
                    d_rot: 1.3012563165005835,
                    sin_theta: 0.9428090415820635,
                    cos_theta: -0.33333333333333315,
                    "task_id": 5094, e_gb: 0.665409}
        for k, v in ans_dict.items():
            if k == 'task_id':
                self.assertEqual(df_gb['task_id'][0], 5094)
            else:
                self.assertAlmostEqual(df_gb[k.str_name][0], v, places=6)

        ans_dict_bulk = {e_coh: 8.301059, G: 147,
                         a0: 2.760381, ar: 1.35,
                         mean_delta_bl: 0,
                         # "breakbond_ratio": 4 / 328,
                         d_gb: 1.0079488085907933,
                         d_rot: 1.3012563165005835,
                         sin_theta: np.sin(0),
                         cos_theta: np.cos(0),
                         "task_id": 5094, e_gb: 0}
        for k, v in ans_dict_bulk.items():
            if k == 'task_id':
                self.assertEqual(df_bulk['task_id'][0], 5094)
            else:
                self.assertAlmostEqual(df_bulk[k.str_name][0], v, places=6)


class TestGBBond(PymatgenTest):
    def setUp(self) -> None:
        self.bulk = Structure.from_file(pjoin(REFS, "mp-91.cif"))
        self.gb_entry = [d for d in load_data() if d["task_id"] == 5094][0]
        self.gb = self.gb_entry['initial_structure']
        self.gbond = GBBond(self.gb, loc_algo='crystalnn')

    def test_loc_algo(self):
        self.assertEqual(len(self.gbond.loc_algo.get_nn_shell_info(structure=self.bulk,
                                                                   site_idx=0,
                                                                   shell=1)),
                         8)
        # With a warning of 'cannot locate an appropriate radius' "

    def test_bond_mat(self):
        bond_mat = self.gbond.bond_mat
        ans = np.array(
            [[0., 2.76269883, 0., 0., 0.,
              0., 0., 0., 0., 2.76270031,
              0., 2.76269957, 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 2.76270031, 0.],
             [2.76269883, 0., 0., 0., 2.76270187,
              0., 0., 2.76270031, 0., 2.76269893,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 0., 0., 2.76269883, 0.,
              0., 2.76269893, 0., 2.76270031, 0.,
              2.7627004, 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 0., 2.76269883, 0., 2.76269883,
              2.76270031, 2.76270031, 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 2.76270187, 0., 2.76269883, 0.,
              2.76269893, 0., 2.76270031, 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 0., 0., 2.76270031, 2.76269893,
              0., 2.60470048, 2.76269883, 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 0., 2.76269893, 2.76270031, 0.,
              2.60470048, 0., 0., 2.76269883, 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 2.76270031, 0., 0., 2.76270031,
              2.76269883, 0., 0., 0., 2.76269883,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 0., 2.76270031, 0., 0.,
              0., 2.76269883, 0., 0., 0.,
              2.76269957, 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 2.76270031],
             [2.76270031, 2.76269893, 0., 0., 0.,
              0., 0., 2.76269883, 0., 0.,
              0., 2.7627004, 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0.],
             [0., 0., 2.7627004, 0., 0.,
              0., 0., 0., 2.76269957, 0.,
              0., 0., 0., 2.76270031, 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 2.7627004],
             [2.76269957, 0., 0., 0., 0.,
              0., 0., 0., 0., 2.7627004,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 2.76270031, 2.7627004, 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 2.76269883, 0.,
              0., 2.76270187, 2.76270031, 2.76269893, 0.,
              0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              2.76270031, 0., 2.76269883, 0., 0.,
              0., 0., 0., 2.76270031, 0.,
              0., 0., 0., 2.76269957],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              2.76269883, 2.76269883, 0., 0., 2.76270031,
              2.76270031, 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 2.76269883,
              0., 0., 0., 0., 2.76269893,
              0., 2.76270031, 2.7627004, 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 2.76270187, 0., 2.76269883,
              0., 0., 2.76270031, 0., 0.,
              2.76269893, 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 2.76270031, 0., 0.,
              0., 2.76270031, 0., 2.76269883, 0.,
              2.76269883, 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 2.76269893, 2.76270031, 0.,
              0., 0., 2.76269883, 0., 0.,
              0., 0., 0., 2.7627004],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 2.76270031,
              2.76269893, 0., 0., 0., 0.,
              2.60470048, 2.76269883, 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 0., 0., 0., 2.76270031,
              0., 2.76269893, 2.76269883, 0., 2.60470048,
              0., 0., 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 2.76270031, 0., 0., 0.,
              2.76270031, 0., 0., 0., 2.76269883,
              0., 0., 2.76269957, 0.],
             [2.76270031, 0., 0., 0., 0.,
              0., 0., 0., 0., 0.,
              0., 2.7627004, 0., 0., 0.,
              2.7627004, 0., 0., 0., 0.,
              0., 2.76269957, 0., 0.],
             [0., 0., 0., 0., 0.,
              0., 0., 0., 2.76270031, 0.,
              2.7627004, 0., 0., 2.76269957, 0.,
              0., 0., 0., 2.7627004, 0.,
              0., 0., 0., 0.]])
        self.assertArrayAlmostEqual(bond_mat, ans)
        dist_mat = self.gb.distance_matrix
        min_bl = (dist_mat[dist_mat > 0]).min()
        self.assertEqual(self.gbond.min_bl, min_bl)

    def test_mean_bl_chg(self):
        b0 = load_b0_dict()['W']
        self.assertEqual(self.gbond.get_mean_bl_chg(b0=b0), -0.004265041421307773)

    def testSerialization(self):
        test = GBBond.from_dict(self.gbond.as_dict())
        self.assertEqual(test.max_bl, self.gbond.max_bl)
        self.assertEqual(test.min_bl, self.gbond.min_bl)
        self.assertArrayAlmostEqual(test.bond_mat, self.gbond.bond_mat)


if __name__ == '__main__':
    unittest.main()
