from pymatgen.util.testing import PymatgenTest

from maml.data.mp import MaterialsProject


class MaterialsProjectTest(PymatgenTest):

    def test_get(self):
        mp = MaterialsProject()
        features = ["pretty_formula", "band_gap", "formation_energy_per_atom", "e_above_hull",
                    "elasticity.K_VRH"]
        df = mp.get(criteria={"nelements": 1},
                    properties=features)
        assert df.shape[0] > 700
        assert df.shape[1] == len(features)
        assert set(df.columns) == set(features)
