from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ..abstract import Describer
import pandas as pd
import numpy as np


class DistinctSiteProperty(Describer):
    """
    Constructs a describer based on properties of distinct sites in a
    structure. For now, this assumes that there is only one type of species in
    a particular Wyckoff site.
    """

    # todo: generalize to multiple sites with the same Wyckoff.

    def fit(self, structures, target=None):
        return self

    def __init__(self, wyckoffs, properties, symprec=0.1):
        """
        :param wyckoffs: List of wyckoff symbols. E.g., ["48a", "24c"]
        :param properties: Sequence of specie properties. E.g.,
            ["atomic_radius"]. Look at pymatgen.core.periodic_table.Element and
            pymatgen.core.periodic_table.Specie for support properties (there
            are a lot!)
        :param symprec: Symmetry precision for spacegroup determination.
        """
        self.wyckoffs = wyckoffs
        self.properties = properties
        self.symprec = symprec

    def describe(self, structure):
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


class CoulombMatrix(Describer):

    def __init__(self, max_sites=None, sorted=False, randomized=False, random_seed=None):

        """
        Coulomb Matrix to decribe structure

        Args:
            max_sites(int): number of max sites,
                if input structure has less site in max_sites, the matrix will be padded to
                the shape of (max_sites, max_sites) with zeros.
            sorted(bool): if True, returns the matrix sorted by the row norm
            randomized(bool): if True, returns the randomized matrix
                              (i) take an arbitrary valid Coulomb matrix C
                              (ii) compute the norm of each row of this Coulomb matrix: row_norms
                              (iii) draw a zero-mean unit-variance noise vector ε of the same
                                    size as row_norms.
                              (iv)  permute the rows and columns of C with the same permutation
                                    that sorts row_norms + ε
                              Montavon, Grégoire, et al.
                              "Machine learning of molecular electronic properties in chemical
                              compound space." New Journal of Physics 15.9 (2013): 095003.
            random_seed(int): random seed

        """

        self.max_sites = None  # For padding
        self.sorted = sorted
        self.randomized = randomized
        self.random_seed = random_seed

    def coulomb_mat(self, s):
        """
        Args
          s(Structure): input structure

        return: np.array
            Coulomb matrix of the structure

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

    def sorted_coulomb_mat(self, s):

        c = self.coulomb_mat(s)
        return c[np.argsort(np.linalg.norm(c, axis=1))]

    def randomized_coulomb_mat(self, s):

        c = self.coulomb_mat(s)
        row_norms = np.linalg.norm(c, axis=1)
        rng = np.random.RandomState(self.random_seed)
        e = rng.normal(size=row_norms.size)
        p = np.argsort(row_norms + e)
        return c[p][:, p]

    def describe(self, structure):
        """
        Args:
            structure(Structure): input structure
        Returns:
            pandas.DataFrame.
            The column is index of the structure, which is 0 for single input
            df[0] returns the serials of coulomb_mat raval
        """
        if self.sorted:
            c = self.sorted_coulomb_mat(structure)
        if self.randomized:
            c = self.randomized_coulomb_mat(structure)
        if np.all([not self.sorted, not self.randomized]):
            c = self.coulomb_mat(structure)
        return pd.DataFrame(c.ravel())

    def describe_all(self, structures):
        """
        Args:
            structures(list): list of Structure

        Returns:
            pandas.DataFrame.
            The columns are the index of input structure in structures
            Indices are the elements index in the coulomb matrix
        """

        return pd.concat([self.describe(s).rename(columns={0: ind})
                          for ind, s in enumerate(structures)], axis=1)
