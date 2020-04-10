from maml.apps import XANES
from maml.apps import CenvPrediction
import unittest
import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr

comp_test_df_path = os.path.join(os.path.dirname(__file__), 'comp_spectra_test.pkl')
comp_test_df = pd.read_pickle(comp_test_df_path)

Al2O3_test_data_path = os.path.join(os.path.dirname(__file__), 'Al2O3_cenv_testdata.pkl')
with open(Al2O3_test_data_path, 'rb') as fp:
    Al2O3_dataset = pickle.load(fp)


class RfxasXANESTest(unittest.TestCase):
    def setUp(self):
        self.test_row = comp_test_df.iloc[0]
        self.test_row_formula = self.test_row['formula']
        self.test_row_ele_group = self.test_row['ele_tm_alka_metalloid']
        self.test_row_xas_id = self.test_row['xas_id']
        self.test_row_absorb_specie = self.test_row['absorbing_species']
        self.test_row_energy_e0 = self.test_row['energy_e0']
        self.test_row_structure = self.test_row['structure']
        self.test_row_x = self.test_row['x_axis_energy_55eV']
        self.test_row_spect = self.test_row['interp_spectrum_55eV']
        self.test_row_x_45 = self.test_row['x_axis_energy_45eV']
        self.test_row_spect_45 = self.test_row['interp_spectrum_45eV']

        self.test_row_add_paras = {
            'composition': self.test_row_formula, 'elemental_group': self.test_row_ele_group,
            'xas_id': self.test_row_xas_id, 'energy_45eV': self.test_row_x_45,
            'spectrum_45eV': self.test_row_spect_45
        }
        self.xanes_obj = XANES(self.test_row_x, self.test_row_spect, self.test_row_absorb_specie,
                               edge='K', e0=self.test_row_energy_e0, **self.test_row_add_paras)

    def test_option_validation(self):
        self.assertRaisesRegex(ValueError, "Invalid energy reference option", CenvPrediction,
                               xanes_spectrum=self.xanes_obj, energy_reference='E1', energy_range=45
                               )
        self.assertRaisesRegex(ValueError, "range needs to be a number", CenvPrediction,
                               xanes_spectrum=self.xanes_obj, energy_reference='lowest', energy_range=[-5, 45]
                               )
        self.assertRaisesRegex(ValueError, "range needs to be larger than 0", CenvPrediction,
                               xanes_spectrum=self.xanes_obj, energy_reference='lowest', energy_range=-20
                               )
        self.assertRaisesRegex(ValueError, "range needs to be a list", CenvPrediction,
                               xanes_spectrum=self.xanes_obj, energy_reference='E0', energy_range=-20,
                               )
        self.assertRaisesRegex(ValueError, "lower bound needs to be less than zero", CenvPrediction,
                               xanes_spectrum=self.xanes_obj, energy_reference='E0', energy_range=[10, 20],
                               )
        self.assertRaisesRegex(ValueError, "higher bound needs to be larger than zero", CenvPrediction,
                               xanes_spectrum=self.xanes_obj, energy_reference='E0', energy_range=[-5, -2],
                               )

    def test_interpolation(self):
        self.xanes_obj_no_interp = CenvPrediction(self.xanes_obj, 'lowest', 45, None, False)
        self.assertTrue(np.allclose(
            pearsonr(self.xanes_obj_no_interp.interp_spectrum, self.xanes_obj_no_interp.xanes_spectrum.y)[0], 1,
            rtol=1e-7))
        self.assertTrue(np.allclose(
            pearsonr(self.xanes_obj_no_interp.interp_energy, self.xanes_obj_no_interp.xanes_spectrum.x)[0], 1,
            rtol=1e-7))

        self.xanes_obj_interp = CenvPrediction(self.xanes_obj, 'lowest', 45, None, True)
        self.assertTrue(
            np.allclose(pearsonr(self.xanes_obj_interp.interp_spectrum, self.xanes_obj.spectrum_45eV)[0], 1, rtol=1e-3))
        self.assertTrue(
            np.allclose(pearsonr(self.xanes_obj_interp.interp_energy, self.xanes_obj.energy_45eV)[0], 1, rtol=1e-5))

    def test_cenv_prediction(self):
        self.Al2O3_origin_obj = XANES(Al2O3_dataset['prev_interp_energy'], Al2O3_dataset['prev_interp_spectrum'],
                                      Al2O3_dataset['absorbing_species'], edge='K', e0=Al2O3_dataset['edge_energy'])
        self.Al2O3_cenv_pred_origin = CenvPrediction(self.Al2O3_origin_obj, 'E0',
                                                     energy_range=Al2O3_dataset['energy_range'],
                                                     edge_energy=Al2O3_dataset['edge_energy'],
                                                     spectrum_interpolation=False)
        self.Al2O3_cenv_pred_origin.cenv_prediction()
        self.assertEqual(self.Al2O3_cenv_pred_origin.pred_cnum_ranklist, 'CN_6')
        self.assertEqual(self.Al2O3_cenv_pred_origin.pred_cenv[0], 'CN_6 coord. motif undetermined')


if __name__ == "__main__":
    unittest.main()
