"""
Structure-wise describers. These describers include structural information.
"""
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd

from maml import BaseDescriber


class _BaseStructDescriber(BaseDescriber):
    def get_type(self):
        return 'structure'


class DistinctSiteProperty(_BaseStructDescriber):
    """
    Constructs a describer based on properties of distinct sites in a
    structure. For now, this assumes that there is only one type of species in
    a particular Wyckoff site.
    """

    # todo: generalize to multiple sites with the same Wyckoff.

    def __init__(self, wyckoffs, properties, symprec=0.1,
                 memory=None, verbose=False, n_jobs=0, **kwargs):
        """

        Args:
            wyckoffs (list of wyckoff symbols):. E.g., ["48a", "24c"]
            properties (list): Sequence of specie properties. E.g.,
                ["atomic_radius"]. Look at pymatgen.core.periodic_table.Element and
                pymatgen.core.periodic_table.Specie for support properties (there
                are a lot!)
            symprec (float): Symmetry precision for spacegroup determination.
            memory (None, str or joblib.Memory): whether to cache to
                the str path
            verbose (bool): whether to show progress for featurization
            n_jobs (int): number of parallel jobs. 0 means no parallel computations.
                If this value is set to negative or greater than the total cpu
                then n_jobs is set to the number of cpu on system
        """
        self.wyckoffs = wyckoffs
        self.properties = properties
        self.symprec = symprec
        super().__init__(memory=memory, verbose=verbose, **kwargs)

    def transform_one(self, structure):
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
        for w in self.wyckoffs:
            site = symm.equivalent_sites[symm.wyckoff_symbols.index(w)][0]
            for p in self.properties:
                data.append(getattr(site.specie, p))
                names.append("%s-%s" % (w, p))
        return pd.DataFrame([data], columns=names)


class CoulombMatrix(_BaseStructDescriber):
    def __init__(self, random_seed=None, memory=None, verbose=False,
                 n_jobs=0, **kwargs):

        """
        Coulomb Matrix to decribe structure

        Args:
            random_seed (int): random seed
            memory (None, str or joblib.Memory): whether to cache to
                the str path
            verbose (bool): whether to show progress for featurization
            n_jobs (int): number of parallel jobs. 0 means no parallel computations.
                If this value is set to negative or greater than the total cpu
                then n_jobs is set to the number of cpu on system
        """

        self.max_sites = None  # For padding
        self.random_seed = random_seed
        super().__init__(memory=memory, verbose=verbose, n_jobs=n_jobs, **kwargs)

    def get_coulomb_mat(self, s):
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

    def transform_one(self, s):
        """
        Args:
            structure(Structure): input structure
            is_sorted (bool): whether to return sorted matrix
            is_randomized (bool): whether to return randomized matrix
        Returns:
            pandas.DataFrame.
            The column is index of the structure, which is 0 for single input
            df[0] returns the serials of coulomb_mat raval
        """
        c = self.get_coulomb_mat(s)
        return pd.DataFrame(c.ravel())


class RandomizedCoulombMatrix(CoulombMatrix):
    """
    Randomized CoulombMatrix
    """
    def __init__(self, random_seed=None, memory=None, verbose=False, n_jobs=0, **kwargs):
        super().__init__(random_seed=random_seed, memory=memory,
                         verbose=verbose, n_jobs=n_jobs, **kwargs)

    def get_randomized_coulomb_mat(self, s):
        """
        Returns the randomized matrix
        (i) take an arbitrary valid Coulomb matrix C
        (ii) compute the norm of each row of this Coulomb matrix: row_norms
        (iii) draw a zero-mean unit-variance noise vector ε of the same
            size as row_norms.
        (iv)  permute the rows and columns of C with the same permutation
            that sorts row_norms + ε
        Montavon, Grégoire, et al.v"Machine learning of molecular electronic properties in chemical
            compound space." New Journal of Physics 15.9 (2013): 095003.

        Args:
            s (pymatgen Structure): pymatgen Structure for computing the randomized Coulomb matrix

        Returns:
            np.ndarray randomized Coulomb matrix
        """
        c = self.get_coulomb_mat(s)
        row_norms = np.linalg.norm(c, axis=1)
        rng = np.random.RandomState(self.random_seed)
        e = rng.normal(size=row_norms.size)
        p = np.argsort(row_norms + e)
        return pd.DataFrame(c[p][:, p].ravel())

    def transform_one(self, s):
        return self.get_randomized_coulomb_mat(s)


class SortedCoulombMatrix(CoulombMatrix):
    """
    Sorted CoulombMatrix
    """
    def __init__(self, random_seed=None, memory=None, verbose=False, n_jobs=0, **kwargs):
        super().__init__(random_seed=random_seed, memory=memory,
                         verbose=verbose, n_jobs=n_jobs, **kwargs)

    def get_sorted_coulomb_mat(self, s):
        """
        Returns the matrix sorted by the row norm

        Args:
            s (pymatgen Structure): pymatgen Structure for computing the Coulomb matrix

        Returns:
            np.ndarray, sorted Coulomb matrix
        """
        c = self.get_coulomb_mat(s)
        return pd.DataFrame(c[np.argsort(np.linalg.norm(c, axis=1))].ravel())

    def transform_one(self, s):
        return self.get_sorted_coulomb_mat(s)
