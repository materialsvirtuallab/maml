"""
Module implements the perturbation class for atomic and lattice relaxation.
"""
import os
from typing import List, Union

import numpy as np
from monty.serialization import loadfn
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup, in_array_list

module_dir = os.path.dirname(__file__)

wyckoff_nums = loadfn(os.path.join(module_dir, "symmetry_rules", "wyckoff_nums.json"))
wyckoff_nums = {int(k): v for k, v in wyckoff_nums.items()}

wyckoff_dims = loadfn(os.path.join(module_dir, "symmetry_rules", "wyckoff_dims.json"))
wyckoff_dims = {int(k): v for k, v in wyckoff_dims.items()}

standard_modes = loadfn(os.path.join(module_dir, "symmetry_rules", "standard_modes.json"))
standard_modes = {int(k): v for k, v in standard_modes.items()}

perturbation_modes = loadfn(os.path.join(module_dir, "symmetry_rules", "perturbation_modes.json"))
perturbation_modes = {int(k): v for k, v in perturbation_modes.items()}

small_addup = np.array([1e-4] * 3)

perturbation_mapping = lambda x, fixed_indices: np.array(
    [
        0 if i in fixed_indices else x[np.argwhere(np.arange(3)[~np.isin(range(3), fixed_indices)] == i)[0][0]]
        for i in range(3)
    ]
)


class WyckoffPerturbation:
    """
    Perturbation class for determining the standard wyckoff position
    and generating corresponding equivalent fractional coordinates.
    """

    def __init__(
        self, int_symbol: int, wyckoff_symbol: str, symmetry_ops: List[SymmOp] = None, use_symmetry: bool = True
    ):
        """
        Args:
            int_symbol (int): International number of space group.
            wyckoff_symbol (str): Wyckoff symbol.
            symmetry_ops (list): Full set of symmetry operations as matrices.
                Use specific symmetry operations if initialized.
            use_symmetry (bool): Whether to use constraint of symmetry to reduce
                parameters space.
        """
        self._site = None
        self._fit_site = False
        self.int_symbol = int_symbol
        self.wyckoff_symbol = wyckoff_symbol
        self.use_symmetry = use_symmetry
        if self.use_symmetry:
            self.standard_mode = eval(standard_modes[int_symbol][wyckoff_symbol])
            self.dim = wyckoff_dims[int_symbol][wyckoff_symbol]
            self.multiplicity = dict(zip(*wyckoff_nums[int_symbol]))[wyckoff_symbol]
            self.perturbation_mode = eval(perturbation_modes[int_symbol][wyckoff_symbol])
            self.symmetry_ops = (
                SpaceGroup.from_int_number(int_symbol).symmetry_ops if not symmetry_ops else symmetry_ops
            )
        else:
            self.standard_mode = eval("lambda p: True")
            self.dim = 3
            self.multiplicity = 1
            self.perturbation_mode = eval("lambda x: x")
            self.symmetry_ops = SpaceGroup.from_int_number(1).symmetry_ops

    def get_orbit(self, p: Union[List, np.ndarray], tol: float = 1e-3) -> List[np.ndarray]:
        """
        Returns the orbit for a point.

        Args:
            p (list/numpy.array): Fractional coordinated point.
            tol (float): Tolerance for determining if sites are the same.
        """
        orbit: List[np.ndarray] = []
        for symm_op in self.symmetry_ops:
            pp = symm_op.operate(p)
            pp[(pp + np.ones(3) * tol) % 1.0 < tol] = 0.0
            pp = np.mod(np.round(pp, decimals=10), 1)
            if not in_array_list(orbit, pp, tol=tol):
                orbit.append(pp)

        return orbit

    def sanity_check(self, site: Union[Site, PeriodicSite], wyc_tol: float = 0.3 * 1e-3) -> None:
        """
        Check whether the perturbation mode exists.

        Args:
            site (PeriodicSite): PeriodicSite in Structure.
            wyc_tol (float): Tolerance for wyckoff symbol determined coordinates.
        """
        p = site.frac_coords
        orbits = self.get_orbit(p, wyc_tol)

        if len(orbits) != self.multiplicity:
            return

        for pp in orbits:
            if self.standard_mode(pp):
                self._site = site  # type: ignore
                self._fit_site = True
                break

    def standardize(self, p: Union[List, np.ndarray], tol: float = 1e-3) -> List[float]:
        """
        Get the standardized position of p.

        Args:
            p (list/numpy.array): Fractional coordinated point.
            tol (float): Tolerance for determining if sites are the same.
        """
        pp: List[float] = []
        orbits = self.get_orbit(p, tol)
        for pp in orbits:  # type: ignore
            if self.standard_mode(pp):
                break
        return pp

    @property
    def site(self):
        """
        Returns the site.
        """
        return self._site

    @property
    def fit_site(self):
        """
        Returns whether the site fits any standard wyckoff position.
        """
        return self._fit_site

    def __repr__(self):
        if self._site is not None:
            return "{}(spg_int_number={}, wyckoff_symbol={}) {} [{:.4f}, {:.4f}, {:.4f}]".format(
                self.__class__.__name__,
                self.int_symbol,
                self.wyckoff_symbol,
                self._site.species_string,
                *self._site.frac_coords,
            )
        return "{}(spg_int_number={}, wyckoff_symbol={})".format(
            self.__class__.__name__, self.int_symbol, self.wyckoff_symbol
        )


def crystal_system(int_number: int) -> str:
    """
    Method for crystal system determination.

    Args:
        int_number (int): International number of space group.
    """
    if int_number <= 2:
        return "triclinic"
    if int_number <= 15:
        return "monoclinic"
    if int_number <= 74:
        return "orthorhombic"
    if int_number <= 142:
        return "tetragonal"
    if int_number <= 167 and int_number not in [
        143,
        144,
        145,
        147,
        149,
        150,
        151,
        152,
        153,
        154,
        156,
        157,
        158,
        159,
        162,
        163,
        164,
        165,
    ]:
        return "rhombohedral"
    if int_number <= 194:
        return "hexagonal"
    return "cubic"


class LatticePerturbation:
    """
    Perturbation class for determining the standard lattice.
    """

    def __init__(self, spg_int_symbol: int, use_symmetry: bool = True):
        """
        Args:
            spg_int_symbol (int): International number of space group.
            use_symmetry (bool): Whether to use constraint of symmetry to reduce
                parameters space.
        """
        self._lattice = None
        self._fit_lattice = False
        self.spg_int_symbol = spg_int_symbol
        self.use_symmetry = use_symmetry
        self.crys_system = crystal_system(spg_int_symbol)

    def sanity_check(self, lattice: Lattice, abc_tol: float = 1e-3, angle_tol: float = 3e-1) -> None:
        """
        Check whether the perturbation mode exists.

        Args:
            lattice (Lattice): Lattice in Structure.
            abc_tol (float): Tolerance for lattice lengths determined by crystal system.
            angle_tol (float): Tolerance for lattice angles determined by crystal system.
        """
        abc = list(lattice.abc)
        angles = list(lattice.angles)

        def check(param, ref, tolerance):
            return all(abs(i - j) < tolerance for i, j in zip(param, ref) if j is not None)

        if not self.use_symmetry:
            self.dims = (3, 3)
            self.perturbation_mode = lambda x: x
            self._lattice = lattice  # type: ignore
            self._abc = abc
            self._fit_lattice = True
            return

        if self.crys_system == "cubic":
            a = abc[0]
            if not (check(abc, [a, a, a], abc_tol) and check(angles, [90, 90, 90], angle_tol)):
                self._fit_lattice = False
                return
            self.dims = (1, 0)
            self.perturbation_mode = lambda x: np.concatenate((np.repeat(x, repeats=3), np.zeros(3)))
            self._lattice = lattice  # type: ignore
            self._abc = [abc[0]]  # type: ignore
            self._fit_lattice = True
            return
        if self.crys_system == "hexagonal":
            if not (
                np.any([(sum(abs(np.array(abc) - abc[i]) < abc_tol) == 2) for i in np.arange(3)])
                and check(np.sort(angles), [90, 90, 120], angle_tol)
            ):
                self._fit_lattice = False
                return
            self.dims = (2, 0)
            indices = [int(sum(abs(np.array(abc) - abc[i]) < abc_tol) == 2) for i in np.arange(3)]
            self.perturbation_mode = lambda x: np.concatenate((x[indices], np.zeros(3)))
            self._lattice = lattice  # type: ignore
            self._abc = [
                abc[indices.index(0)],  # type: ignore
                abc[indices.index(1)],
            ]  # type: ignore
            self._fit_lattice = True
            return
        if self.crys_system == "rhombohedral":
            a = abc[0]
            alpha = angles[0]
            if check(abc, [a, a, a], abc_tol) and check(angles, [alpha, alpha, alpha], angle_tol):
                self.dims = (1, 1)
                self.perturbation_mode = lambda x: np.concatenate(
                    (np.repeat(x[0], repeats=3), np.repeat(x[1], repeats=3))
                )
                self._lattice = lattice  # type: ignore
                self._abc = [a]  # type: ignore
                self._fit_lattice = True
                return
            if np.any([(sum(abs(np.array(abc) - abc[i]) < abc_tol) == 2) for i in np.arange(3)]) and check(
                np.sort(angles), [90, 90, 120], angle_tol
            ):
                self.dims = (2, 0)
                indices = [int(sum(abs(np.array(abc) - abc[i]) < abc_tol) == 2) for i in np.arange(3)]
                self.perturbation_mode = lambda x: np.concatenate((x[indices], np.zeros(3)))
                self._lattice = lattice  # type: ignore
                self._abc = [
                    abc[indices.index(0)],  # type: ignore
                    abc[indices.index(1)],
                ]  # type: ignore
                self._fit_lattice = True
                return
            self._fit_lattice = False
            return
        if self.crys_system == "tetragonal":
            if not check(angles, [90, 90, 90], angle_tol):
                self._fit_lattice = False
                return
            if np.any([(sum(abs(np.array(abc) - abc[i]) < abc_tol) == 2) for i in np.arange(3)]):
                self.dims = (2, 0)
                indices = [int(sum(abs(np.array(abc) - abc[i]) < abc_tol) == 2) for i in np.arange(3)]
                self.perturbation_mode = lambda x: np.concatenate((x[indices], np.zeros(3)))
                self._lattice = lattice  # type: ignore
                self._abc = [
                    abc[indices.index(0)],  # type: ignore
                    abc[indices.index(1)],
                ]  # type: ignore
                self._fit_lattice = True
                return
            if np.all([(sum(abs(np.array(abc) - abc[i]) < abc_tol) == 3) for i in np.arange(3)]):
                self.dims = (1, 0)
                self.perturbation_mode = lambda x: np.concatenate((np.repeat(x, repeats=3), np.zeros(3)))
                self._lattice = lattice  # type: ignore
                self._abc = [abc[0]]  # type: ignore
                self._fit_lattice = True
                return
            self._fit_lattice = False
            return
        if self.crys_system == "orthorhombic":
            if not check(angles, [90, 90, 90], angle_tol):
                self._fit_lattice = False
                return
            self.dims = (3, 0)
            self.perturbation_mode = lambda x: np.concatenate((x, np.zeros(3)))
            self._lattice = lattice  # type: ignore
            self._abc = list(abc)  # type: ignore
            self._fit_lattice = True
            return
        if self.crys_system == "monoclinic":
            if sum(abs(angles[i] - 90) < 1e-5 for i in np.arange(3)) == 2:
                self.dims = (3, 1)
                indices = [int(abs(angles[i] - 90) < angle_tol) for i in np.arange(3)]
                self.perturbation_mode = lambda x: np.concatenate(
                    (
                        np.pad(x, pad_width=(0, 5 - len(x)), mode="constant")[:3],
                        np.pad(x, pad_width=(0, 5 - len(x)), mode="constant")[3:][indices],
                    )
                )
                self._lattice = lattice  # type: ignore
                self._abc = list(abc)  # type: ignore
                self._fit_lattice = True
                return
            if check(angles, [90, 90, 90], angle_tol):
                self.dims = (3, 0)
                self.perturbation_mode = lambda x: np.concatenate((x, np.zeros(3)))
                self._lattice = lattice  # type: ignore
                self._abc = list(abc)  # type: ignore
                self._fit_lattice = True
                return
            self._fit_lattice = False
            return
        self.dims = (3, 3)
        self.perturbation_mode = lambda x: x
        self._lattice = lattice  # type: ignore
        self._abc = list(abc)  # type: ignore
        self._fit_lattice = True
        return

    @property
    def fit_lattice(self) -> bool:
        """
        Returns whether the lattice fits any crystal system.
        """
        return self._fit_lattice

    @property
    def lattice(self) -> Lattice:
        """
        Returns the lattice.
        """
        return self._lattice  # type: ignore

    @property
    def abc(self) -> List[float]:
        """
        Returns the lattice lengths.
        """
        return self._abc  # type: ignore

    def __repr__(self):
        if self._lattice is not None:
            return "{}(spg_int_number={}, crystal_system={})\n".format(
                self.__class__.__name__, self.spg_int_symbol, self.crys_system
            ) + repr(self.lattice)
        return "{}(spg_int_number={}, crystal_system={})\n".format(
            self.__class__.__name__, self.spg_int_symbol, self.crys_system
        )


def get_standardized_structure(structure: Structure) -> Structure:
    """
    Get standardized structure.

    Args:
        structure (Structure): Pymatgen Structure object.
    """
    species = [dict(site.species.as_dict()) for site in structure]
    sa = SpacegroupAnalyzer(structure)
    sd = sa.get_symmetry_dataset()
    std_lattice = sd["std_lattice"]
    mapping_to_primitive = sd["mapping_to_primitive"].tolist()
    mapping_to_primitive = np.array([mapping_to_primitive.index(i) for i in range(max(mapping_to_primitive) + 1)])
    std_species = np.array(species)[mapping_to_primitive][sd["std_mapping_to_primitive"]]
    std_frac_coords = sd["std_positions"]
    return Structure(lattice=std_lattice, species=std_species, coords=std_frac_coords).get_sorted_structure()
