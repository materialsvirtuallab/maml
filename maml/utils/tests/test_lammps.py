from __future__ import annotations

import unittest

import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Lattice, Structure
from pymatgen.io.lammps.data import LammpsData

from maml.utils._lammps import (
    check_structures_forces_stresses,
    stress_format_change,
    stress_list_to_matrix,
    write_data_from_structure,
)


class TestLAMMPS(unittest.TestCase):
    def test_stress_format_convert(self):
        vasp = ["xx", "yy", "zz", "xy", "yz", "xz"]
        lammps = stress_format_change(vasp, "VASP", "LAMMPS")
        self.assertListEqual(lammps.tolist(), ["xx", "yy", "zz", "xy", "xz", "yz"])
        snap = stress_format_change(vasp, "VASP", "SNAP")
        self.assertListEqual(snap.tolist(), ["xx", "yy", "zz", "yz", "xz", "xy"])

    def test_stress_list_matrix(self):
        a = [1, 2, 3, 4, 5, 6]
        matrix = stress_list_to_matrix(a, "VASP")
        matrix = matrix.tolist()
        self.assertListEqual(matrix[0], [1, 4, 6])
        self.assertListEqual(matrix[1], [4, 2, 5])
        self.assertListEqual(matrix[2], [6, 5, 3])

    def test_check_structures(self):
        s = Structure(
            Lattice(np.array([[3.16, 0.1, 0.2], [0.1, 3.17, 0.3], [0.1, 0.2, 3]])),
            ["Mo", "Mo"],
            [[0, 0, 0], [0.13, 0.4, 0.2]],
        )

        forces = np.array([[0.04844841, 0.08648062, 0.07070806], [-0.04844841, -0.08648062, -0.07070806]])

        stress = np.array([-0.22279327, -1.2809575, -0.44279698, -0.23345818, -0.37798718, -0.17676364])

        checked_force = np.array([[0.05552151, 0.09063424, 0.05940176], [-0.05552151, -0.09063424, -0.05940176]])

        checked_stress = np.array([-0.26319715, -1.3219795, -0.3613719, -0.30627516, -0.27276486, -0.17306383])

        new_structures, new_forces, new_stresses = check_structures_forces_stresses([s], [forces], [stress])

        # print(np.linalg.norm(checked_stress - new_stresses[0]))
        print(new_stresses[0], checked_stress)
        assert np.linalg.norm(checked_force - new_forces[0]) < 0.0001
        assert np.linalg.norm(checked_stress - new_stresses[0]) < 0.0001

        new_structures = check_structures_forces_stresses(structures=[s], return_none=False)

        assert len(new_structures) == 1
        assert isinstance(new_structures[0], Structure)

        new_structures, new_forces, new_stresses = check_structures_forces_stresses(structures=[s, s], return_none=True)

        assert len(new_forces) == 2
        assert new_forces[0] is None
        assert len(new_stresses) == 2
        assert new_stresses[0] is None

    def test_write(self):
        s = Structure(
            Lattice(np.array([[3.16, 0.1, 0.2], [0.1, 3.17, 0.3], [0.1, 0.2, 3]])),
            ["Mo", "Mo"],
            [[0, 0, 0], [0.13, 0.4, 0.2]],
        )
        with ScratchDir("."):
            write_data_from_structure(s, "test.data")
            lmp = LammpsData.from_file("test.data", atom_style="charge")
            lmp2 = LammpsData.from_structure(s)
            lmp2.write_file("test2.data")
            assert str(lmp) == str(lmp2)


if __name__ == "__main__":
    unittest.main()
