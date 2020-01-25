# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import itertools
import subprocess
import re
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.os.path import which
from monty.tempfile import ScratchDir
from pymatgen.core.periodic_table import get_el_sp

from maml.abstract import Describer
from maml.apps.pes.processing import pool_from


class BispectrumCoefficients(Describer):
    """
    Bispectrum coefficients to describe the local environment of each
    atom in a quantitative way.

    """

    def __init__(self, rcutfac, twojmax, element_profile, rfac0=0.99363,
                 rmin0=0, diagonalstyle=3, quadratic=False, pot_fit=False):
        """

        Args:
            rcutfac (float): Global cutoff distance.
            twojmax (int): Band limit for bispectrum components.
            element_profile (dict): Parameters (cutoff factor 'r' and
                weight 'w') related to each element, e.g.,
                {'Na': {'r': 0.3, 'w': 0.9},
                 'Cl': {'r': 0.7, 'w': 3.0}}
            rfac0 (float): Parameter in distance to angle conversion.
                Set between (0, 1), default to 0.99363.
            rmin0 (float): Parameter in distance to angle conversion.
                Default to 0.
            diagonalstyle (int): Parameter defining which bispectrum
                components are generated. Choose among 0, 1, 2 and 3,
                default to 3.
            quadratic (bool): Whether including quadratic terms.
                Default to False.
            pot_fit (bool): Whether to output in potential fitting
                format. Default to False, i.e., returning the bispectrum
                coefficients for each site.

        """
        from maml.apps.pes.lammps.calcs import SpectralNeighborAnalysis
        self.calculator = SpectralNeighborAnalysis(rcutfac, twojmax,
                                                   element_profile,
                                                   rfac0, rmin0,
                                                   diagonalstyle,
                                                   quadratic)
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.element_profile = element_profile
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        self.diagonalstyle = diagonalstyle
        self.elements = sorted(element_profile.keys(),
                               key=lambda sym: get_el_sp(sym).X)
        self.quadratic = quadratic
        self.pot_fit = pot_fit

    @property
    def subscripts(self):
        """
        The subscripts (2j1, 2j2, 2j) of all bispectrum components
        involved.

        """
        return self.calculator.get_bs_subscripts(self.twojmax,
                                                 self.diagonalstyle)

    def describe(self, structure, include_stress=False):
        """
        Returns data for one input structure.

        Args:
            structure (Structure): Input structure.
            include_stress (bool): Whether to include stress descriptors.

        Returns:
            DataFrame.

            In regular format, the columns are the subscripts of
            bispectrum components, while indices are the site indices
            in input structure.

            In potential fitting format, to match the sequence of
            [energy, f_x[0], f_y[0], ..., f_z[N], v_xx, ..., v_xy], the
            bispectrum coefficients are summed up by each specie and
            normalized by a factor of No. of atoms (in the 1st row),
            while the derivatives in each direction are preserved, with
            the columns being the subscripts of bispectrum components
            with each specie and the indices being
            [0, '0_x', '0_y', ..., 'N_z'], and the virial contributions
            (in GPa) are summed up for all atoms for each component in
            the sequence of ['xx', 'yy', 'zz', 'yz', 'xz', 'xy'].

        """
        return self.describe_all([structure], include_stress).xs(0, level='input_index')

    def describe_all(self, structures, include_stress=False):
        """
        Returns data for all input structures in a single DataFrame.

        Args:
            structures (Structure): Input structures as a list.
            include_stress (bool): Whether to include stress descriptors.

        Returns:
            DataFrame with indices of input list preserved. To retrieve
            the data for structures[i], use
            df.xs(i, level='input_index').

        """
        columns = list(map(lambda s: '-'.join(['%d' % i for i in s]),
                           self.subscripts))
        if self.quadratic:
            columns += list(map(lambda s: '-'.join(['%d%d%d' % (i, j, k)
                                                    for i, j, k in s]),
                                itertools.combinations_with_replacement(self.subscripts, 2)))

        raw_data = self.calculator.calculate(structures)

        def process(output, combine, idx, include_stress):
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
                dbs = np.hstack([np.insert(d.reshape(-1, len(columns)),
                                           0, 0, axis=1) for d in dbs])
                db_index = ['%d_%s' % (i, d)
                            for i in df_b.index for d in 'xyz']
                df_db = pd.DataFrame(dbs, index=db_index,
                                     columns=hstack_b.columns)
                if include_stress:
                    vbs = np.split(vb.sum(axis=0), len(self.elements))
                    vbs = np.hstack([np.insert(v.reshape(-1, len(columns)),
                                               0, 0, axis=1) for v in vbs])
                    volume = structures[idx].volume
                    vbs = vbs / volume * 160.21766208  # from eV to GPa
                    vb_index = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
                    df_vb = pd.DataFrame(vbs, index=vb_index,
                                         columns=hstack_b.columns)
                    df = pd.concat([hstack_b, df_db, df_vb])
                else:
                    df = pd.concat([hstack_b, df_db])
            return df

        df = pd.concat([process(d, self.pot_fit, i, include_stress)
                        for i, d in enumerate(raw_data)],
                       keys=range(len(raw_data)), names=["input_index", None])
        return df


class AGNIFingerprints(Describer):
    """
    Fingerprints for AGNI (Adaptive, Generalizable and Neighborhood
    Informed) force field. Elemental systems only.

    """

    def __init__(self, r_cut, etas):
        """

        Args:
            r_cut (float): Cutoff distance.
            etas (numpy.array): All eta parameters in 1D array.
        """
        self.r_cut = r_cut
        self.etas = etas

    def describe(self, structure):
        """
        Calculate fingerprints for all sites in a structure.

        Args:
            structure (Structure): Input structure.

        Returns:
            DataFrame.

        """
        all_neighbors = structure.get_all_neighbors(self.r_cut)
        fingerprints = []
        for i, an in enumerate(all_neighbors):
            center = structure[i].coords
            coords, distances = zip(*[(site.coords, d) for (site, d) in an])
            v = (np.array(coords) - center)[:, :, None]
            d = np.array(distances)[:, None, None]
            e = np.array(self.etas)[None, None, :]
            cf = 0.5 * (np.cos(np.pi * d / self.r_cut) + 1)
            fpi = np.sum(v / d * np.exp(-(d / e) ** 2) * cf, axis=0)
            fingerprints.append(fpi)
        index = ["%d_%s" % (i, d) for i in range(len(structure))
                 for d in "xyz"]
        df = pd.DataFrame(np.vstack(fingerprints), index=index,
                          columns=self.etas)
        return df

    def describe_all(self, structures):
        return pd.concat([self.describe(s) for s in structures],
                         keys=range(len(structures)),
                         names=['input_index', None])


class SOAPDescriptor(Describer):
    """
    Smooth Overlap of Atomic Position (SOAP) descriptor.
    """

    def __init__(self, cutoff, l_max=8, n_max=8, atom_sigma=0.5):
        """

        Args:
            cutoff (float): Cutoff radius.
            l_max (int): The band limit of spherical harmonics basis function.
                Default to 8.
            n_max (int): The number of radial basis function. Default to 8.
            atom_sigma (float): The width of gaussian atomic density.
                Default to 0.5.
        """
        from maml.potential.soap import SOAPotential

        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.atom_sigma = atom_sigma
        self.operator = SOAPotential()

    def describe(self, structure):
        """
        Returns data for one input structure.

        Args:
            structure (Structure): Input structure.
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
        atomic_numbers = [str(num) for num in np.unique(structure.atomic_numbers)]
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

    def describe_all(self, structures):
        return pd.concat([self.describe(s) for s in structures],
                         keys=range(len(structures)),
                         names=['input_index', None])


class BPSymmetryFunctions(Describer):
    """
    Behler-Parrinello symmetry function descriptor.
    """

    def __init__(self, dmin, cutoff, num_symm2, a_etas):
        """
        Args:
            dmin (float): The minimum interatomic distance accepted.
            cutoff (float): Cutoff radius.
            num_symm2 (int): The number of radial symmetry functions.
            a_etas (list): The choice of η' in angular symmetry functions.
        """

        from maml.potential.nnp import NNPotential

        self.dmin = dmin
        self.cutoff = cutoff
        self.num_symm2 = num_symm2
        self.a_etas = a_etas
        self.operator = NNPotential()

    def describe(self, structure):
        """
        Returns data for one input structure.

        Args:
            structure (Structure): Input structure.
        """
        if not which('RuNNer'):
            raise RuntimeError("RuNNer has not been found.")
        if not which("RuNNerMakesym"):
            raise RuntimeError("RuNNerMakesym has not been found.")

        def read_functions_data(filename):
            """
            Read structure features from file.

            Args:
                filename (str): The functions file to be read.
            """
            with zopen(filename, 'rt') as f:
                lines = f.read()

            block_pattern = re.compile(r'(\n\s+\d+\n|^\s+\d+\n)(.+?)(?=\n\s+\d+\n|$)', re.S)
            points_features = []
            for (num_neighbor, block) in block_pattern.findall(lines):
                point_features = pd.DataFrame([feature.split()[1:]
                                               for feature in block.split('\n')[:-1]],
                                              dtype=np.float32)
                points_features.append(point_features)
            points_features = pd.concat(points_features,
                                        keys=range(len(block_pattern.findall(lines))),
                                        names=['point_index', None])
            return points_features

        dmin = sorted(set(structure.distance_matrix.ravel()))[1]
        r_etas = self.operator.generate_eta(dmin=self.dmin,
                                            r_cut=self.cutoff,
                                            num_symm2=self.num_symm2)
        atoms_filename = 'input.data'
        mode_output = 'mode.out'

        with ScratchDir('.'):
            atoms_filename = self.operator.write_cfgs(filename=atoms_filename,
                                                      cfg_pool=pool_from([structure]))
            input_filename = self.operator.write_input(mode=1, r_cut=self.cutoff,
                                                       r_etas=r_etas, a_etas=self.a_etas,
                                                       scale_feature=False)
            p = subprocess.Popen(['RuNNer'], stdout=open(mode_output, 'w'))
            stdout = p.communicate()[0]

            descriptors = read_functions_data('function.data')

        return pd.DataFrame(descriptors)

    def describe_all(self, structures):
        return pd.concat([self.describe(s) for s in structures],
                         keys=range(len(structures)),
                         names=['input_index', None])
