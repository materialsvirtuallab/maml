from __future__ import annotations

import unittest

import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest

from maml.describers import RadialDistributionFunction


class TestRDF(PymatgenTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rdf_instance = RadialDistributionFunction(r_max=4)
        cls.structure = cls.get_structure("SrTiO3")

    def test_get_site_rdf(self):
        r, site_rdfs = self.rdf_instance.get_site_rdf(self.structure)
        r_ref = np.linspace(self.rdf_instance.r_min, self.rdf_instance.r_max, self.rdf_instance.n_grid)
        assert np.allclose(r, r_ref)
        assert "Sr2+:O2-" in site_rdfs[0]
        assert "Sr2+:Sr2+" in site_rdfs[0]
        assert "Sr2+:Ti4+" in site_rdfs[0]

        a = self.structure.lattice.a
        sr_o = np.sqrt(2) / 2.0 * a
        sr_sr = a
        sr_ti = np.sqrt(3) / 2.0 * a
        sr_o_cal = r[np.argmax(site_rdfs[0]["Sr2+:O2-"])]
        sr_sr_cal = r[np.argmax(site_rdfs[0]["Sr2+:Sr2+"])]
        sr_ti_cal = r[np.argmax(site_rdfs[0]["Sr2+:Ti4+"])]
        assert sr_o_cal == pytest.approx(sr_o, 2)
        assert sr_sr_cal == pytest.approx(sr_sr, 1)
        assert sr_ti_cal == pytest.approx(sr_ti, 1)

    def test_get_species_rdf(self):
        r, rdf = self.rdf_instance.get_species_rdf(self.structure)
        assert self.structure.lattice.a * np.sqrt(2) / 2.0 == pytest.approx(r[np.argmax(rdf)], 2)

    def test_get_site_cn(self):
        r, cns = self.rdf_instance.get_site_coordination(self.structure)
        assert cns[0]["Sr2+:O2-"][-1].item() == pytest.approx(12)
        assert cns[0]["Sr2+:Ti4+"][-1].item() == pytest.approx(8)
        assert cns[0]["Sr2+:Sr2+"][-1].item() == pytest.approx(6)

    def test_get_specie_cn(self):
        r, cn = self.rdf_instance.get_species_coordination(self.structure)
        assert cn[-1] == 106


if __name__ == "__main__":
    unittest.main()
