"""
Structure-wise describers. These describers include structural information.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

from maml import BaseDescriber
from ._megnet import MEGNetStructure


__all__ = ['DistinctSiteProperty', 'MEGNetStructure', 'CoulombMatrix',
           'RandomizedCoulombMatrix', 'SortedCoulombMatrix']


class DistinctSiteProperty(BaseDescriber):
    """
    Constructs a describer based on properties of distinct sites in a
    structure. For now, this assumes that there is only one type of species in
    a particular Wyckoff site.
    """
    # todo: generalize to multiple sites with the same Wyckoff.
    supported_properties = ["mendeleev_no", "electrical_resistivity",
                            "velocity_of_sound", "reflectivity",
                            "refractive_index", "poissons_ratio", "molar_volume",
                            "thermal_conductivity", "boiling_point", "melting_point",
                            "critical_temperature", "superconduction_temperature",
                            "liquid_range", "bulk_modulus", "youngs_modulus",
                            "brinell_hardness", "rigidity_modulus",
                            "mineral_hardness", "vickers_hardness",
                            "density_of_solid", "atomic_radius_calculated",
                            "van_der_waals_radius", "coefficient_of_linear_thermal_expansion",
                            "ground_state_term_symbol", "valence", "Z", "X",
                            "atomic_mass", "block", "row", "group", "atomic_radius",
                            "average_ionic_radius", "average_cationic_radius",
                            "average_anionic_radius", "metallic_radius", "ionic_radii",
                            "oxi_state", "max_oxidation_state", "min_oxidation_state",
                            "is_transition_metal", "is_alkali", "is_alkaline", "is_chalcogen",
                            "is_halogen", "is_lanthanoid", "is_metal", "is_metalloid",
                            "is_noble_gas", "is_post_transition_metal", "is_quadrupolar",
                            "is_rare_earth_metal", "is_actinoid"]

    def __init__(self,
                 properties: List[str],
                 symprec: float = 0.1,
                 wyckoffs: Optional[List[str]] = None,
                 feature_batch: str = "pandas_concat",
                 **kwargs):
        """

        Args:
            properties (list): Sequence of specie properties. E.g.,
                ["atomic_radius"]. Look at pymatgen.core.periodic_table.Element and
                pymatgen.core.periodic_table.Specie for support properties (there
                are a lot!)
            symprec (float): Symmetry precision for spacegroup determination.
            wyckoffs (list of wyckoff symbols):. E.g., ["48a", "24c"], if not provided,
                will get from input structure when doing transform
            feature_batch (str): way to batch a list of features into one
            **kwargs: keyword args to specify memory, verbose, and n_jobs
        """
        self.properties = properties
        self.symprec = symprec
        self.wyckoffs = wyckoffs
        super().__init__(feature_batch=feature_batch, **kwargs)

    def transform_one(self, structure: Structure) -> pd.DataFrame:
        """

        Args:
            structure (pymatgen Structure): pymatgen structure for descriptor computation

        Returns:
            pd.DataFrame that contains the distinct position labeled features

        """
        a = SpacegroupAnalyzer(structure, self.symprec)
        symm = a.get_symmetrized_structure()
        data = []
        names = []
        if self.wyckoffs is None:
            self.wyckoffs = symm.wyckoff_symbols
        for w in self.wyckoffs:
            site = symm.equivalent_sites[symm.wyckoff_symbols.index(w)][0]
            for p in self.properties:
                data.append(getattr(site.specie, p))
                names.append("%s-%s" % (w, p))
        return pd.DataFrame([data], columns=names)

    def get_citations(self) -> List[str]:
        """
        Get distinct site property citations
        """
        return ["@article{ye2018deep,"
                "title={Deep neural networks for accurate predictions of crystal stability},"
                "author={Ye, Weike and Chen, Chi and Wang, Zhenbin and "
                "Chu, Iek-Heng and Ong, Shyue Ping},"
                "journal={Nature communications},"
                "volume={9},"
                "number={1},"
                "pages={1--6},"
                "year={2018},"
                "publisher={Nature Publishing Group}}"]


class CoulombMatrix(BaseDescriber):
    """
    Coulomb Matrix to decribe structure

    """
    def __init__(self,
                 random_seed: int = None,
                 **kwargs):
        """
        Args:
            random_seed (int): random seed
            **kwargs: keyword args to specify memory, verbose, and n_jobs
        """

        self.max_sites = None  # For padding
        self.random_seed = random_seed
        if 'feature_batch' not in kwargs:
            kwargs['feature_batch'] = 'pandas_concat'
        super().__init__(**kwargs)

    def get_coulomb_mat(self, s: Structure) -> np.ndarray:
        """
        Args:
            s (pymatgen Structure): input structure

        Returns:
            np.ndarray Coulomb matrix of the structure

        """
        dis = s.distance_matrix
        num_sites = s.num_sites
        c = np.zeros((num_sites, num_sites))

        for i in range(num_sites):
            for j in range(num_sites):
                if i == j:
                    c[i, j] = 0.5 * (s[i].specie.Z ** 2.4)

                elif i < j:
                    c[i, j] = (s[i].specie.Z * s[j].specie.Z) / dis[i, j]
                    c[j, i] = c[i, j]

                else:
                    continue

        if self.max_sites and self.max_sites > num_sites:
            padding = self.max_sites - num_sites
            return np.pad(c, (0, padding),
                          mode='constant',
                          constant_values=0)

        return c

    def transform_one(self, s: Structure) -> pd.DataFrame:
        """
        Args:
            s (Structure): input structure

        Returns:
            pandas.DataFrame.
            The column is index of the structure, which is 0 for single input
            df[0] returns the serials of coulomb_mat raval
        """
        c = self.get_coulomb_mat(s)
        return pd.DataFrame(c.ravel())

    def get_citations(self) -> List[str]:
        """
        Citations for CoulombMatrix
        """
        return ["@article{rupp2012fast, "
                "title={Fast and accurate modeling of molecular "
                "atomization energies with machine learning},"
                "author={Rupp, Matthias and Tkatchenko, Alexandre and M{\"u}ller, "
                "Klaus-Robert and Von Lilienfeld, O Anatole},"
                "journal={Physical review letters}, volume={108}, "
                "number={5}, pages={058301}, "
                "year={2012}, publisher={APS}}"]


class RandomizedCoulombMatrix(CoulombMatrix):
    """
    Randomized CoulombMatrix
    """
    def __init__(self,
                 random_seed: int = None,
                 **kwargs):
        """
        Args:
            random_seed (int): random seed
            **kwargs: keyword args to specify memory, verbose, and n_jobs
        """
        super().__init__(random_seed=random_seed, **kwargs)

    def get_randomized_coulomb_mat(self, s: Structure) -> pd.DataFrame:
        """
        Returns the randomized matrix
        (i) take an arbitrary valid Coulomb matrix C
        (ii) compute the norm of each row of this Coulomb matrix: row_norms
        (iii) draw a zero-mean unit-variance noise vector ε of the same
            size as row_norms.
        (iv)  permute the rows and columns of C with the same permutation
            that sorts row_norms + ε

        Args:
            s (pymatgen Structure): pymatgen Structure for computing the randomized Coulomb matrix

        Returns:
            pd.DataFrame randomized Coulomb matrix
        """
        c = self.get_coulomb_mat(s)
        row_norms = np.linalg.norm(c, axis=1)
        rng = np.random.RandomState(self.random_seed)
        e = rng.normal(size=row_norms.size)
        p = np.argsort(row_norms + e)
        return pd.DataFrame(c[p][:, p].ravel())

    def transform_one(self, s: Structure) -> pd.DataFrame:
        """
        Transform one structure to descriptors
        Args:
            s (Structure): pymatgen structure

        Returns: pandas dataframe descriptors

        """
        return self.get_randomized_coulomb_mat(s)

    def get_citations(self) -> List[str]:
        """
        citation for randomized coulomb matrix
        """
        return ["@article{montavon2013machine,"
                "title={Machine learning of molecular electronic properties "
                "in chemical compound space},"
                "author={Montavon, Gr{\'e}goire and Rupp, Matthias and Gobre, "
                "Vivekanand and Vazquez-Mayagoitia, Alvaro and Hansen, Katja "
                "and Tkatchenko, Alexandre and M{\"u}ller, Klaus-Robert and "
                "Von Lilienfeld, O Anatole},"
                "journal={New Journal of Physics},"
                "volume={15}, number={9},pages={095003},"
                "year={2013},publisher={IOP Publishing}}"]


class SortedCoulombMatrix(CoulombMatrix):
    """
    Sorted CoulombMatrix
    """
    def __init__(self,
                 random_seed: int = None,
                 **kwargs):
        """
        Args:
            random_seed (int): random seed
            **kwargs: keyword args to specify memory, verbose, and n_jobs
        """
        super().__init__(random_seed=random_seed, **kwargs)

    def get_sorted_coulomb_mat(self, s: Structure) -> pd.DataFrame:
        """
        Returns the matrix sorted by the row norm

        Args:
            s (pymatgen Structure): pymatgen Structure for computing the Coulomb matrix

        Returns:
            pd.DataFrame, sorted Coulomb matrix
        """
        c = self.get_coulomb_mat(s)
        return pd.DataFrame(c[np.argsort(np.linalg.norm(c, axis=1))].ravel())

    def transform_one(self, s: Structure) -> pd.DataFrame:
        """
        Transform one structure into descriptor
        Args:
            s (Structure): pymatgen Structure

        Returns: pd.DataFrame descriptors

        """
        return self.get_sorted_coulomb_mat(s)

    def get_citations(self) -> List[str]:
        """
        Sorted Coulomb matrix
        """
        return ["@inproceedings{montavon2012learning,"
                "title={Learning invariant representations "
                "of molecules for atomization energy prediction},"
                "author={Montavon, Gr{\'e}goire and Hansen, Katja "
                "and Fazli, Siamac and Rupp, Matthias and Biegler, "
                "Franziska and Ziehe, Andreas and Tkatchenko, Alexandre "
                "and Lilienfeld, Anatole V and M{\"u}ller, Klaus-Robert},"
                "booktitle={Advances in neural information processing systems},"
                "pages={440--448}, year={2012}}"]
