# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import os
import abc
import io
import subprocess
import itertools

import numpy as np
from monty.tempfile import ScratchDir
from pymatgen import Element
from pymatgen.io.lammps.data import LammpsData
from maml.apps.pes.abstract import Potential


def _sort_elements(symbols):
    return [e.symbol for e in sorted([Element(e) for e in symbols])]


def _pretty_input(lines):
    clean_lines = [l.strip('\n') for l in lines]
    commands = [l for l in clean_lines if len(l.strip()) > 0]
    keys = [c.split()[0] for c in commands
            if not c.split()[0].startswith('#')]
    width = max([len(k) for k in keys]) + 4

    def prettify(l):
        return l.split()[0].ljust(width) + ' '.join(l.split()[1:]) \
            if not (len(l.split()) == 0 or l.strip().startswith('#')) else l
    new_lines = map(prettify, clean_lines)
    return '\n'.join(new_lines)


def _read_dump(file_name, dtype='float_'):
    with open(file_name) as f:
        lines = f.readlines()[9:]
    return np.loadtxt(io.StringIO(''.join(lines)), dtype=dtype)


class LMPStaticCalculator(metaclass=abc.ABCMeta):
    """
    Abstract class to perform static structure property calculation
    using LAMMPS.

    """

    LMP_EXE = 'lmp_serial'
    _COMMON_CMDS = ['units metal',
                    'atom_style charge',
                    'box tilt large',
                    'read_data data.static',
                    'run 0']

    @abc.abstractmethod
    def _setup(self):
        """
        Setup a calculation, writing input files, etc.

        """
        return

    @abc.abstractmethod
    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return

    @abc.abstractmethod
    def _parse(self):
        """
        Parse results from dump files.

        """
        return

    def calculate(self, structures):
        """
        Perform the calculation on a series of structures.

        Args:
            structures [Structure]: Input structures in a list.

        Returns:
            List of computed data corresponding to each structure,
            varies with different subclasses.

        """
        for s in structures:
            assert self._sanity_check(s) is True, \
                'Incompatible structure found'
        ff_elements = None
        if hasattr(self, 'element_profile'):
            ff_elements = self.element_profile.keys()
        with ScratchDir('.'):
            input_file = self._setup()
            data = []
            for s in structures:
                ld = LammpsData.from_structure(s, ff_elements)
                ld.write_file('data.static')
                p = subprocess.Popen([self.LMP_EXE, '-in', input_file],
                                     stdout=subprocess.PIPE)
                stdout = p.communicate()[0]
                rc = p.returncode
                if rc != 0:
                    error_msg = 'LAMMPS exited with return code %d' % rc
                    msg = stdout.decode("utf-8").split('\n')[:-1]
                    try:
                        error_line = [i for i, m in enumerate(msg)
                                      if m.startswith('ERROR')][0]
                        error_msg += ', '.join([e for e in msg[error_line:]])
                    except Exception:
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)
                results = self._parse()
                data.append(results)
        return data


class EnergyForceStress(LMPStaticCalculator):
    """
    Calculate energy, forces and virial stress of structures.
    """

    def __init__(self, ff_settings):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
        """
        self.ff_settings = ff_settings

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'efs')
        with open(os.path.join(template_dir, 'in.efs'), 'r') as f:
            input_template = f.read()

        input_file = 'in.efs'

        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))
        return input_file

    def _sanity_check(self, structure):
        return True

    def _parse(self):
        energy = float(np.loadtxt('energy.txt'))
        force = _read_dump('force.dump')
        stress = np.loadtxt('stress.txt')
        return energy, force, stress


class SpectralNeighborAnalysis(LMPStaticCalculator):
    """
    Calculator for bispectrum components to characterize the local
    neighborhood of each atom in a general way.

    Usage:
        [(b, db, e)] = sna.calculate([Structure])
        b: 2d NumPy array with shape (N, n_bs) containing bispectrum
            coefficients, where N is the No. of atoms in structure and
            n_bs is the No. of bispectrum components.
        db: 2d NumPy array with shape (N, 3 * n_bs * n_elements)
            containing the first order derivatives of bispectrum
            coefficients with respect to atomic coordinates,
            where n_elements is the No. of elements in element_profile.
        e: 2d NumPy array with shape (N, 1) containing the element of
            each atom.

    """

    _CMDS = ['pair_style lj/cut 10',
             'pair_coeff * * 1 1',
             'compute sna all sna/atom ',
             'compute snad all snad/atom ',
             'compute snav all snav/atom ',
             'dump 1 all custom 1 dump.element element',
             'dump 2 all custom 1 dump.sna c_sna[*]',
             'dump 3 all custom 1 dump.snad c_snad[*]',
             'dump 4 all custom 1 dump.snav c_snav[*]']

    def __init__(self, rcutfac, twojmax, element_profile, rfac0=0.99363,
                 rmin0=0, diagonalstyle=3, quadratic=False):
        """
        For more details on the parameters, please refer to the
        official documentation of LAMMPS.

        Notes:
            Despite this calculator uses compute sna(d)/atom command
            (http://lammps.sandia.gov/doc/compute_sna_atom.html), the
            parameter definition is in consistent with pair_style snap
            document (http://lammps.sandia.gov/doc/pair_snap.html),
            where *rcutfac* is the cutoff in distance rather than some
            scale factor.

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

        """
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.element_profile = element_profile
        self.rfac0 = rfac0
        self.rmin0 = rmin0
        assert diagonalstyle in range(4), 'Invalid diagonalstype, ' \
                                          'choose among 0, 1, 2 and 3'
        self.diagonalstyle = diagonalstyle
        self.quadratic = quadratic

    @staticmethod
    def get_bs_subscripts(twojmax, diagonal):
        """
        Method to list the subscripts 2j1, 2j2, 2j of bispectrum
        components.

        Args:
            twojmax (int): Band limit for bispectrum components.
            diagonal (int): Parameter defining which bispectrum
            components are generated. Choose among 0, 1, 2 and 3.

        Returns:
            List of all subscripts [2j1, 2j2, 2j].

        """
        subs = itertools.product(range(twojmax + 1), repeat=3)
        filters = [lambda x: x[0] >= x[1]]
        if diagonal == 2:
            filters.append(lambda x: x[0] == x[1] == x[2])
        else:
            if diagonal == 1:
                filters.append(lambda x: x[0] == x[1])
            elif diagonal == 3:
                filters.append(lambda x: x[2] >= x[0])
            elif diagonal == 0:
                pass

            def j_filter(x):
                return x[2] in range(x[0] - x[1], min(twojmax, x[0] + x[1]) + 1, 2)
            filters.append(j_filter)
        for f in filters:
            subs = filter(f, subs)
        return list(subs)

    @property
    def n_bs(self):
        """
        Returns No. of bispectrum components to be calculated.

        """
        return len(self.get_bs_subscripts(self.twojmax, self.diagonalstyle))

    def _setup(self):
        compute_args = '{} {} {} '.format(1, self.rfac0, self.twojmax)
        el_in_seq = _sort_elements(self.element_profile.keys())
        cutoffs = [self.element_profile[e]['r'] * self.rcutfac
                   for e in el_in_seq]
        weights = [self.element_profile[e]['w'] for e in el_in_seq]
        compute_args += ' '.join([str(p) for p in cutoffs + weights])
        qflag = 1 if self.quadratic else 0
        compute_args += ' diagonal {} rmin0 {} quadraticflag {}'. \
            format(self.diagonalstyle, self.rmin0, qflag)

        def add_args(l):
            return l + compute_args if l.startswith('compute') else l
        CMDS = list(map(add_args, self._CMDS))
        CMDS[2] += ' bzeroflag 0'
        CMDS[3] += ' bzeroflag 0'
        CMDS[4] += ' bzeroflag 0'
        dump_modify = 'dump_modify 1 element '
        dump_modify += ' '.join(str(e) for e in el_in_seq)
        CMDS.append(dump_modify)
        ALL_CMDS = self._COMMON_CMDS[:]
        ALL_CMDS[-1:-1] = CMDS
        input_file = 'in.sna'
        with open('in.sna', 'w') as f:
            f.write(_pretty_input(ALL_CMDS).format(self.twojmax, self.rfac0))
        return input_file

    def _sanity_check(self, structure):
        struc_elements = set(structure.symbol_set)
        sna_elements = self.element_profile.keys()
        return struc_elements.issubset(sna_elements)

    def _parse(self):
        element = np.atleast_1d(_read_dump('dump.element', 'unicode'))
        b = np.atleast_2d(_read_dump('dump.sna'))
        db = np.atleast_2d(_read_dump('dump.snad'))
        vb = np.atleast_2d(_read_dump('dump.snav'))
        return b, db, vb, element


class ElasticConstant(LMPStaticCalculator):
    """
    Elastic constant calculator.
    """
    _RESTART_CONFIG = {'internal': {'write_command': 'write_restart',
                                    'read_command': 'read_restart',
                                    'restart_file': 'restart.equil'},
                       'external': {'write_command': 'write_data',
                                    'read_command': 'read_data',
                                    'restart_file': 'data.static'}}

    def __init__(self, ff_settings, potential_type='external',
                 deformation_size=1e-6, jiggle=1e-5, lattice='bcc', alat=5.0,
                 maxiter=400, maxeval=1000):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            potential_type (str): 'internal' indicates the internal potential
                installed in lammps, 'external' indicates the external potential
                outside of lammps.
            deformation_size (float): Finite deformation size. Usually range from
                1e-2 to 1e-8, to confirm the results not depend on it.
            jiggle (float): The amount of random jiggle for atoms to
                prevent atoms from staying on saddle points.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            maxiter (float): The maximum number of iteration. Default to 400.
            maxeval (float): The maximum number of evaluation. Default to 1000.
        """
        self.ff_settings = ff_settings
        self.write_command = self._RESTART_CONFIG[potential_type]['write_command']
        self.read_command = self._RESTART_CONFIG[potential_type]['read_command']
        self.restart_file = self._RESTART_CONFIG[potential_type]['restart_file']
        self.deformation_size = deformation_size
        self.jiggle = jiggle
        self.lattice = lattice
        self.alat = alat
        self.maxiter = maxiter
        self.maxeval = maxeval

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'elastic')

        with open(os.path.join(template_dir, 'in.elastic'), 'r') as f:
            input_template = f.read()
        with open(os.path.join(template_dir, 'init.template'), 'r') as f:
            init_template = f.read()
        with open(os.path.join(template_dir, 'potential.template'), 'r') as f:
            potential_template = f.read()
        with open(os.path.join(template_dir, 'displace.template'), 'r') as f:
            displace_template = f.read()

        input_file = 'in.elastic'

        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(write_restart=self.write_command,
                                          restart_file=self.restart_file))
        with open('init.mod', 'w') as f:
            f.write(init_template.format(deformation_size=self.deformation_size,
                                         jiggle=self.jiggle, maxiter=self.maxiter,
                                         maxeval=self.maxeval, lattice=self.lattice,
                                         alat=self.alat))
        with open('potential.mod', 'w') as f:
            f.write(potential_template.format(ff_settings='\n'.join(ff_settings)))
        with open('displace.mod', 'w') as f:
            f.write(displace_template.format(read_restart=self.read_command,
                                             restart_file=self.restart_file))
        return input_file

    def calculate(self):
        with ScratchDir('.'):
            input_file = self._setup()
            p = subprocess.Popen([self.LMP_EXE, '-in', input_file],
                                 stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            rc = p.returncode
            if rc != 0:
                error_msg = 'LAMMPS exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            result = self._parse()
        return result

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        C11, C12, C44, bulkmodulus = np.loadtxt('elastic.txt')
        return C11, C12, C44, bulkmodulus


class LatticeConstant(LMPStaticCalculator):
    """
    Lattice Constant Relaxation Calculator.
    """

    def __init__(self, ff_settings):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
        """
        self.ff_settings = ff_settings

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'latt')

        with open(os.path.join(template_dir, 'in.latt'), 'r') as f:
            input_template = f.read()

        input_file = 'in.latt'

        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))

        return input_file

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump files.

        """
        a, b, c = np.loadtxt('lattice.txt')
        return a, b, c


class TimeBenchmarker(LMPStaticCalculator):
    """
    Time benchmark calculator.
    """

    def __init__(self, ff_settings):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
        """
        self.ff_settings = ff_settings

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'time_benchmark')

        with open(os.path.join(template_dir, 'in.time'), 'r') as f:
            input_template = f.read()

        input_file = 'in.time'

        if isinstance(self.ff_settings, Potential):
            ff_settings = self.ff_settings.write_param()
        else:
            ff_settings = self.ff_settings

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings)))

        return input_file

    def _sanity_check(self, structure):
        """
        Check if the structure is valid for this calculation.

        """
        return True

    def _parse(self):
        """
        Parse results from dump file.

        """
        return 0
