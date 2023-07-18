from __future__ import annotations

import os
import unittest

from pymatgen.util.testing import PymatgenTest

import maml.apps.gbe as gbe
from maml.apps.gbe.utils import load_b0_dict, load_data, load_mean_delta_bl_dict, pjoin

module_dir = os.path.dirname(gbe.__file__)
REFS = pjoin(module_dir, "references")
DATA = pjoin(module_dir, "data")


class TestUtil(PymatgenTest):
    def test_load_data(self):
        data = load_data()
        assert len(data) == 361

    def test_b0(self):
        b0_dict = load_b0_dict()
        assert os.path.isfile(pjoin(REFS, "el2b0.json"))
        assert b0_dict["W"] == 2.7603814963781725
        els = [
            "Ac",
            "Ag",
            "Al",
            "Au",
            "Ba",
            "Be",
            "Ca",
            "Cd",
            "Ce",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "Dy",
            "Er",
            "Eu",
            "Fe",
            "Gd",
            "Hf",
            "Hg",
            "Ho",
            "Ir",
            "K",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mo",
            "Na",
            "Nb",
            "Nd",
            "Ni",
            "Os",
            "Pb",
            "Pd",
            "Pm",
            "Pr",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Ru",
            "Sc",
            "Sm",
            "Sr",
            "Ta",
            "Tb",
            "Tc",
            "Th",
            "Ti",
            "Tl",
            "Tm",
            "V",
            "W",
            "Y",
            "Yb",
            "Zn",
            "Zr",
        ]
        assert sorted(b0_dict.keys()) == els

    def test_mean_delta_bl(self):
        mean_delta_bl = load_mean_delta_bl_dict()
        assert os.path.isfile(pjoin(REFS, "mean_bl_chg_crystalnn.json"))
        assert mean_delta_bl["5094"] == -0.004265041421307773
        assert len(mean_delta_bl) == 361


if __name__ == "__main__":
    unittest.main()
