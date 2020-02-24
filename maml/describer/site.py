"""
This module provides local environment describers.
"""
import re
import logging
import itertools
import subprocess
import numpy as np
import pandas as pd

from monty.io import zopen
from monty.os.path import which
from monty.tempfile import ScratchDir

from pymatgen import Element
from pymatgen.core.periodic_table import get_el_sp
from maml.base.describer import BaseDescriber, OutDataFrameConcat
from maml.utils.data_conversion import pool_from


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BispectrumCoefficients(OutDataFrameConcat, BaseDescriber):
    """
    Bispectrum coefficients to describe the local environment of each atom.
    Lammps is required to perform this computation.
    """
    def __init__(self, cutoff, twojmax, element_profile, quadratic=False,
                 pot_fit=False, include_stress=False,
                 memory=None, verbose=False, n_jobs=0):
        """
        Args:
            cutoff (float): The cutoff distance.
            twojmax (int): Band limit for bispectrum components.
            element_profile (dict): Parameters (cutoff factor 'r' and weight 'w')
                related to each element, e.g.,
                {'Na': {'r': 0.3, 'w': 0.9},
                 'Cl': {'r': 0.7, 'w': 3.0}}
            quadratic (bool): Whether including quadratic terms.
                Default to False.
            pot_fit (bool): Whether combine the dataframe for potential fitting.
            include_stress (bool): Wether to include stress components.
            memory (str/joblib.Memory): Whether to cache to the str path.
            verbose (bool): Whether to show progress for featurization.
            n_jobs (int): number of parallel jobs. 0 means no parallel computations.
                If this value is set to negative or greater than the total cpu
                then n_jobs is set to the number of cpu on system.
        """
        from maml.apps.pes.lammps.calcs import SpectralNeighborAnalysis
        self.calculator = SpectralNeighborAnalysis(rcut=cutoff,
                                                   twojmax=twojmax,
                                                   element_profile=element_profile,
                                                   quadratic=quadratic)
        self.rcutfac = cutoff
        self.twojmax = twojmax
        self.elements = sorted(element_profile.keys(), key=lambda x: Element(x))
        self.element_profile = element_profile
        self.quadratic = quadratic
        self.pot_fit = pot_fit
        self.include_stress = include_stress
        super().__init__(memory=memory, verbose=verbose, n_jobs=n_jobs)

    @property
    def subscripts(self):
        """
        The subscripts (2j1, 2j2, 2j) of all bispectrum components
        involved.
        """
        return self.calculator.get_bs_subscripts(self.twojmax)

    def transform_one(self, structure):
        """
        Args:
            structure (Structure): Pymatgen Structure object.
        """
        columns = list(map(lambda s: '-'.join(['%d' % i for i in s]), self.subscripts))
        if self.quadratic:
            columns += list(map(lambda s: '-'.join(['%d%d%d' % (i, j, k) for i, j, k in s]),
                                itertools.combinations_with_replacement(self.subscripts, 2)))

        raw_data = self.calculator.calculate([structure])

        def process(output, combine):
            b, db, vb, e = output
            df = pd.DataFrame(b, columns=columns)
            if combine:
                df_add = pd.DataFrame({'element': e, 'n': np.ones(len(e))})
                df_b = df_add.join(df)
                n_atoms = df_b.shape[0]
                b_by_el = [df_b[df_b['element'] == e] for e in self.elements]
                sum_b = [df[df.columns[1:]].sum(axis=0) for df in b_by_el]
                hstack_b = pd.concat(sum_b, keys=self.elements)
                hstack_b = hstack_b.to_frame().T / n_atoms
                hstack_b.fillna(0, inplace=True)
                dbs = np.split(db, len(self.elements), axis=1)
                dbs = np.hstack([np.insert(d.reshape(-1, len(columns)), 0, 0, axis=1)
                                 for d in dbs])
                db_index = ['%d_%s' % (i, d) for i in df_b.index for d in 'xyz']
                df_db = pd.DataFrame(dbs, index=db_index, columns=hstack_b.columns)
                if self.include_stress:
                    vbs = np.split(vb.sum(axis=0), len(self.elements))
                    vbs = np.hstack([np.insert(v.reshape(-1, len(columns)),
                                               0, 0, axis=1) for v in vbs])
                    volume = structure.volume
                    vbs = vbs / volume * 160.21766208  # from eV to GPa
                    vb_index = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
                    df_vb = pd.DataFrame(vbs, index=vb_index, columns=hstack_b.columns)
                    df = pd.concat([hstack_b, df_db, df_vb])
                else:
                    df = pd.concat([hstack_b, df_db])
            return df

        return process(raw_data[0], self.pot_fit)


class SmoothOverlapAtomicPosition(OutDataFrameConcat, BaseDescriber):
    """
    Smooth overlap of atomic positions (SOAP) to describe the local environment
    of each atom.
    """
    def __init__(self, cutoff, l_max=8, n_max=8, atom_sigma=0.5, memory=None,
                 verbose=False, n_jobs=0):
        """

        Args:
            cutoff (float): Cutoff radius.
            l_max (int): The band limit of spherical harmonics basis function.
                Default to 8.
            n_max (int): The number of radial basis function. Default to 8.
            atom_sigma (float): The width of gaussian atomic density. Default to 0.5.
            memory (str/joblib.Memory): Whether to cache to the str path.
            verbose (bool): Whether to show progress for featurization.
            n_jobs (int): number of parallel jobs. 0 means no parallel computations.
                If this value is set to negative or greater than the total cpu
                then n_jobs is set to the number of cpu on system.
        """
        from maml.apps.pes.gap import GAPotential
        self.operator = GAPotential()
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.atom_sigma = atom_sigma
        super().__init__(memory=memory, verbose=verbose, n_jobs=n_jobs)

    def transform_one(self, structure):
        """
        Args:
            structure (Structure): Pymatgen Structure object.
        """
        if not which('quip'):
            raise RuntimeError("quip has not been found.\n",
                               "Please refer to https://github.com/libAtoms/QUIP for ",
                               "further detail.")

        atoms_filename = 'structure.xyz'

        exe_command = ['quip']
        exe_command.append('atoms_filename={}'.format(atoms_filename))

        descriptor_command = ['soap']
        descriptor_command.append("cutoff" + '=' + '{}'.format(self.cutoff))
        descriptor_command.append("l_max" + '=' + '{}'.format(self.l_max))
        descriptor_command.append("n_max" + '=' + '{}'.format(self.n_max))
        descriptor_command.append("atom_sigma" + '=' + '{}'.format(self.atom_sigma))

        atomic_numbers = [str(element.number) for element in sorted(np.unique(structure.species))]
        n_Z = len(atomic_numbers)
        n_species = len(atomic_numbers)
        Z = '{' + '{}'.format(' '.join(atomic_numbers)) + '}'
        species_Z = '{' + '{}'.format(' '.join(atomic_numbers)) + '}'
        descriptor_command.append("n_Z" + '=' + str(n_Z))
        descriptor_command.append("Z" + '=' + Z)
        descriptor_command.append("n_species" + '=' + str(n_species))
        descriptor_command.append("species_Z" + '=' + species_Z)

        exe_command.append("descriptor_str=" + "{" +
                           "{}".format(' '.join(descriptor_command)) + "}")

        with ScratchDir('.'):
            atoms_filename = self.operator.write_cfgs(filename=atoms_filename,
                                                      cfg_pool=pool_from([structure]))
            descriptor_output = 'output'
            p = subprocess.Popen(exe_command, stdout=open(descriptor_output, 'w'))
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'QUIP exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            with zopen(descriptor_output, 'rt') as f:
                lines = f.read()

            descriptor_pattern = re.compile('DESC(.*?)\n', re.S)
            descriptors = pd.DataFrame([np.array(c.split(), dtype=np.float)
                                        for c in descriptor_pattern.findall(lines)])

        return descriptors


class BPSymmetryFunctions(OutDataFrameConcat, BaseDescriber):
    """
    Behler-Parrinello symmetry function to describe the local environment
    of each atom.
    """
    def __init__(self, cutoff, r_etas, r_shift, a_etas, zetas, lambdas,
                 memory=None, verbose=False, n_jobs=0):
        """
        Args:
            cutoff (float): The cutoff distance.
            r_etas (numpy.ndarray): η in radial function.
            r_shift (numpy.ndarray): Rs in radial function.
            a_etas (numpy.ndarray): η in angular function.
            zetas (numpy.ndarray): ζ in angular function.
            lambdas (numpy.ndarray): λ in angular function. Default to (1, -1).
        """
        self.cutoff = cutoff
        self.r_etas = np.array(r_etas)[None, :, None]
        self.r_shift = np.array(r_shift)[None, None, :]
        self.a_etas = np.array(a_etas)[None, :, None, None]
        self.zetas = np.array(zetas)[None, None, :, None]
        self.lambdas = np.array(lambdas)[None, None, None, :]
        super().__init__(memory=memory, verbose=verbose, n_jobs=n_jobs)

    def transform_one(self, structure):
        """
        Args:
            structure (Structure): Pymatgen Structure object.
        """
        elements = sorted(structure.symbol_set, key=lambda sym: get_el_sp(sym).X)
        all_neighbors = structure.get_all_neighbors(self.cutoff)
        data = []

        for atom_idx, neighbors in enumerate(all_neighbors):
            center = structure[atom_idx].coords
            site_symmfuncs = []
            sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbor.species_string)
            temp_dict = {element: [(neighbor.coords, neighbor.nn_distance) for neighbor in group]
                         for element, group in itertools.groupby(sorted_neighbors,
                         key=lambda neighbor: neighbor.species_string)}

            for specie in elements:
                distances = np.array([nn_distance for _, nn_distance 
                                      in temp_dict[specie]])[:, None, None]
                g2 = np.sum(np.exp(-self.r_etas * (distances - self.r_shift) ** 2)
                            * self._fc(distances), axis=0)
                site_symmfuncs.extend(g2.ravel().tolist())

            for specie1, specie2 in itertools.combinations_with_replacement(elements, 2):

                group1, group2 = temp_dict[specie1], temp_dict[specie2]

                c = itertools.combinations(range(len(group1)), 2) if specie1 == specie2 \
                    else itertools.product(range(len(group1)), range(len(group2)))
                index_combination = np.array(list(c)).T

                coords_group1 = np.array([coords for coords, _ in group1])
                coords_group2 = np.array([coords for coords, _ in group2])
                distances_group1 = np.array([nn_distance for _, nn_distance in group1])
                distances_group2 = np.array([nn_distance for _, nn_distance in group2])

                v1 = coords_group1[index_combination[0]]
                v2 = coords_group2[index_combination[1]]
                d1 = distances_group1[index_combination[0]]
                d2 = distances_group2[index_combination[1]]
                d3 = np.linalg.norm(v1 - v2, axis=1)
                cosines = np.sum((v1 - center) * (v2 - center), axis=1) / (d1 * d2)
                cosines = cosines[:, None, None, None]
                distances = np.stack((d1, d2, d3))
                cutoffs = np.prod(self._fc(distances), axis=0)
                cutoffs = np.atleast_1d(cutoffs)[:, None, None, None]
                powers = np.sum(distances ** 2, axis=0)[:, None, None, None]
                g4 = np.sum((1 + self.lambdas * cosines) ** self.zetas *
                            np.exp(-self.a_etas * powers) * cutoffs *
                            2.0 ** (1 - self.zetas), axis=0)
                site_symmfuncs.extend(g4.ravel().tolist())

            data.append(site_symmfuncs)
        df = pd.DataFrame(data)
        return df

    def _fc(self, r):
        """
        Cutoff function to decay the symmetry functions at vicinity of radial cutoff.

        Args:
            r (float): The pair distance.
        """
        decay = 0.5 * (np.cos(np.pi * r / self.cutoff) + 1) * (r <= self.cutoff)
        return decay
