import unittest
import os
import numpy as np

from pymatgen.util.testing import PymatgenTest
from pymatgen.core import Structure

from maml.apps.bowsr.model.cgcnn import CGCNN

module_dir = os.path.dirname(__file__)
pjoin = os.path.join


class TestCGCNN(PymatgenTest):
    def test_predict_energy(self):
        structure = Structure.from_file(pjoin(module_dir, "mp-6812.cif"))
        cgcnn = CGCNN()
        output = cgcnn.predict_energy(structure)
        self.assertAlmostEqual(output.astype(np.float64), -3.0786154)


if __name__ == "__main__":
    unittest.main()
