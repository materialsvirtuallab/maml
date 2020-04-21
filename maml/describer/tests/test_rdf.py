import unittest

import numpy as np
from pymatgen.util.testing import PymatgenTest

from maml.describer._rdf import RadialDistributionFunction


class TestRDF(PymatgenTest):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rdf_instance = RadialDistributionFunction(r_max=4)
        cls.structure = cls.get_structure("SrTiO3")

    def test_get_site_rdf(self):
        r, site_rdfs = self.rdf_instance.get_site_rdf(self.structure)
        r_ref = np.linspace(self.rdf_instance.r_min, self.rdf_instance.r_max,
                            self.rdf_instance.n_grid)
        self.assertArrayAlmostEqual(r, r_ref)
        self.assertIn('Sr2+:O2-', site_rdfs[0])
        self.assertIn('Sr2+:Sr2+', site_rdfs[0])
        self.assertIn('Sr2+:Ti4+', site_rdfs[0])

        a = self.structure.lattice.a
        sr_o = np.sqrt(2) / 2. * a
        sr_sr = a
        sr_ti = np.sqrt(3) / 2. * a
        sr_o_cal = r[np.argmax(site_rdfs[0]['Sr2+:O2-'])]
        sr_sr_cal = r[np.argmax(site_rdfs[0]['Sr2+:Sr2+'])]
        sr_ti_cal = r[np.argmax(site_rdfs[0]['Sr2+:Ti4+'])]
        self.assertAlmostEqual(sr_o_cal, sr_o, 2)
        self.assertAlmostEqual(sr_sr_cal, sr_sr, 1)
        self.assertAlmostEqual(sr_ti_cal, sr_ti, 1)

    def test_get_species_rdf(self):
        r, rdf = self.rdf_instance.get_species_rdf(self.structure)
        self.assertAlmostEqual(self.structure.lattice.a * np.sqrt(2) / 2.,
                               r[np.argmax(rdf)], 2)

    def test_get_site_cn(self):
        r, cns = self.rdf_instance.get_site_coordination(self.structure)
        self.assertAlmostEqual(cns[0]['Sr2+:O2-'][-1].item(), 12)
        self.assertAlmostEqual(cns[0]['Sr2+:Ti4+'][-1].item(), 8)
        self.assertAlmostEqual(cns[0]['Sr2+:Sr2+'][-1].item(), 6)

    def test_get_specie_cn(self):
        r, cn = self.rdf_instance.get_species_coordination(self.structure)
        self.assertAlmostEqual(cn[-1], 106)


if __name__ == "__main__":
    unittest.main()
