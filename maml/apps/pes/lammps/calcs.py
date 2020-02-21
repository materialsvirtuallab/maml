# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""This module provides basic LAMMPS calculator classes."""

import os
import abc
import io
import subprocess
import itertools

import numpy as np
from monty.tempfile import ScratchDir

from pymatgen.io.lammps.data import LammpsData
from pymatgen import Structure, Lattice, Element

from maml.apps.pes import Potential


def _pretty_input(lines):
    clean_lines = [l.strip('\n') for l in lines]
    commands = [l for l in clean_lines if len(l.strip()) > 0]
    keys = [c.split()[0] for c in commands if not c.split()[0].startswith('#')]
    width = max([len(k) for k in keys]) + 4
    prettify = lambda l: l.split()[0].ljust(width) + ' '.join(l.split()[1:]) \
        if not (len(l.split()) == 0 or l.strip().startswith('#')) else l
    new_lines = map(prettify, clean_lines)
    return '\n'.join(new_lines)


def _read_dump(file_name, dtype='float_'):
    with open(file_name) as f:
        lines = f.readlines()[9:]
    return np.loadtxt(io.StringIO(''.join(lines)), dtype=dtype)


class LMPStaticCalculator(object):
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
        for struct in structures:
            assert self._sanity_check(struct) is True, \
                'Incompatible structure found'
        ff_elements = None
        if hasattr(self, 'element_profile'):
            ff_elements = self.element_profile.keys()
        if hasattr(self, 'ff_settings') and hasattr(self.ff_settings, 'elements'):
            ff_elements = self.ff_settings.elements
        with ScratchDir('.'):
            input_file = self._setup()
            data = []
            for struct in structures:
                ld = LammpsData.from_structure(struct, ff_elements)
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

    def __init__(self, rcut, twojmax, element_profile, quadratic=False):
        """
        For more details on the parameters, please refer to the
        official documentation of LAMMPS.

        Notes:
            Despite this calculator uses compute sna(d)/atom command
            (http://lammps.sandia.gov/doc/compute_sna_atom.html), the
            parameter definition is in consistent with pair_style snap
            document (http://lammps.sandia.gov/doc/pair_snap.html).

        Args:
            rcut (float): The cutoff distance.
            twojmax (int): Band limit for bispectrum components.
            element_profile (dict): Parameters (cutoff factor 'r' and
                weight 'w') related to each element, e.g.,
                {'Na': {'r': 0.3, 'w': 0.9},
                 'Cl': {'r': 0.7, 'w': 3.0}}
            quadratic (bool): Whether including quadratic terms.
                Default to False.

        """
        self.rcutfac = rcut
        self.twojmax = twojmax
        self.element_profile = element_profile
        self.quadratic = quadratic

    @staticmethod
    def get_bs_subscripts(twojmax):
        """
        Method to list the subscripts 2j1, 2j2, 2j of bispectrum
        components.

        Args:
            twojmax (int): Band limit for bispectrum components.

        Returns:
            List of all subscripts [2j1, 2j2, 2j].

        """
        subs = itertools.product(range(twojmax + 1), repeat=3)
        filters = [lambda x: True if x[0] >= x[1] else False,
                   lambda x: True if x[2] >= x[0] else False]
        j_filter = lambda x: True if x[2] in range(x[0] - x[1],
                            min(twojmax, x[0] + x[1]) + 1, 2) else False
        filters.append(j_filter)
        for f in filters:
            subs = filter(f, subs)
        return list(subs)

    @property
    def n_bs(self):
        """
        Returns No. of bispectrum components to be calculated.

        """
        return len(self.get_bs_subscripts(self.twojmax))

    def _setup(self):
        compute_args = '1 0.99363 {} '.format(self.twojmax)
        el_in_seq = sorted(self.element_profile.keys(), key=lambda x: Element(x))
        cutoffs = [self.element_profile[e]['r'] * self.rcutfac for e in el_in_seq]
        weights = [self.element_profile[e]['w'] for e in el_in_seq]
        compute_args += ' '.join([str(p) for p in cutoffs + weights])
        compute_args += ' diagonal 3 rmin0 0 quadraticflag {}'.format(int(self.quadratic))
        add_args = lambda l: l + compute_args if l.startswith('compute') else l
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
        with open(input_file, 'w') as f:
            f.write(_pretty_input(ALL_CMDS))
        return input_file

    def _sanity_check(self, structure):
        elements = set(structure.symbol_set)
        sna_elements = self.element_profile.keys()
        return elements.issubset(sna_elements)

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
            potential_type (str): 'internal' indicates the internal potentials
                installed in lammps, 'external' indicates the external potentials
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
        """
        Calculate the elastic constant given Potential class.
        """
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


class NudgedElasticBand(LMPStaticCalculator):
    """
    NudgedElasticBand migration energy calculator.
    """

    def __init__(self, ff_settings, specie, lattice, alat, num_replicas=7):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            num_replicas (int): Number of replicas to use.
        """
        self.ff_settings = ff_settings
        self.specie = specie
        self.lattice = lattice
        self.alat = alat
        self.num_replicas = num_replicas

    def get_unit_cell(self, specie, lattice, alat):
        """
        Get the unit cell from specie, lattice type and lattice constant.

        Args
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        if lattice == 'fcc':
            unit_cell = Structure.from_spacegroup(sg='Fm-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'bcc':
            unit_cell = Structure.from_spacegroup(sg='Im-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'diamond':
            unit_cell = Structure.from_spacegroup(sg='Fd-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        else:
            raise ValueError("Lattice type is invalid.")

        return unit_cell

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'neb')

        with open(os.path.join(template_dir, 'in.relax'), 'r') as f:
            relax_template = f.read()
        with open(os.path.join(template_dir, 'in.neb'), 'r') as f:
            neb_template = f.read()

        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=self.alat)
        lattice_calculator = LatticeConstant(ff_settings=self.ff_settings)
        a, _, _ = lattice_calculator.calculate([unit_cell])[0]
        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=a)

        if self.lattice == 'fcc':
            start_idx, final_idx = 95, 49
            scale_factor = [3, 3, 3]
        elif self.lattice == 'bcc':
            start_idx, final_idx = 40, 14
            scale_factor = [3, 3, 3]
        elif self.lattice == 'diamond':
            start_idx, final_idx = 7, 15
            scale_factor = [2, 2, 2]
        else:
            raise ValueError("Lattice type is invalid.")

        super_cell = unit_cell * scale_factor
        super_cell_ld = LammpsData.from_structure(super_cell, atom_style='atomic')
        super_cell_ld.write_file('data.supercell')

        with open('in.relax', 'w') as f:
            f.write(relax_template.format(ff_settings='\n'.join(self.ff_settings.write_param()),
                                          lattice=self.lattice, alat=a, specie=self.specie,
                                          del_id=start_idx + 1, relaxed_file='initial.relaxed'))

        p = subprocess.Popen([self.LMP_EXE, '-in', 'in.relax'], stdout=subprocess.PIPE)
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

        with open('in.relax', 'w') as f:
            f.write(relax_template.format(ff_settings='\n'.join(self.ff_settings.write_param()),
                                          lattice=self.lattice, alat=a, specie=self.specie,
                                          del_id=final_idx + 1, relaxed_file='final.relaxed'))

        p = subprocess.Popen([self.LMP_EXE, '-in', 'in.relax'], stdout=subprocess.PIPE)
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

        final_relaxed_struct = LammpsData.from_file('final.relaxed',
                                                    atom_style='atomic').structure

        lines = ['{}'.format(final_relaxed_struct.num_sites)]

        for idx, site in enumerate(final_relaxed_struct):
            if idx == final_idx:
                idx = final_relaxed_struct.num_sites
            elif idx == start_idx:
                idx = final_idx
            else:
                idx = idx
            lines.append('{}  {:.3f}  {:.3f}  {:.3f}'.format(idx + 1, site.x, site.y, site.z))

        with open('data.final_replica', 'w') as f:
            f.write('\n'.join(lines))

        input_file = 'in.neb'

        with open(input_file, 'w') as f:
            f.write(neb_template.format(ff_settings='\n'.join(self.ff_settings.write_param()),
                                        start_replica='initial.relaxed',
                                        final_replica='data.final_replica'))

        return input_file

    def calculate(self):
        """
        Calculate the NEB barrier given Potential class.
        """
        with ScratchDir('.'):
            input_file = self._setup()
            p = subprocess.Popen(['mpirun', '-n', str(self.num_replicas),
                                  'lmp_mpi', '-partition', '{}x1'.format(self.num_replicas),
                                  '-in', input_file],
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
        with open('log.lammps') as f:
            lines = f.readlines()[-1:]
        migration_barrier = float(lines[0].split()[6])
        return migration_barrier


class DefectFormation(LMPStaticCalculator):
    """
    Defect formation energy calculator.
    """

    def __init__(self, ff_settings, specie, lattice, alat):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        self.ff_settings = ff_settings
        self.specie = specie
        self.lattice = lattice
        self.alat = alat

    def get_unit_cell(self, specie, lattice, alat):
        """
        Get the unit cell from specie, lattice type and lattice constant.

        Args
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        if lattice == 'fcc':
            unit_cell = Structure.from_spacegroup(sg='Fm-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'bcc':
            unit_cell = Structure.from_spacegroup(sg='Im-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        elif lattice == 'diamond':
            unit_cell = Structure.from_spacegroup(sg='Fd-3m',
                                                  lattice=Lattice.cubic(alat),
                                                  species=[specie], coords=[[0, 0, 0]])
        else:
            raise ValueError("Lattice type is invalid.")

        return unit_cell

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates', 'defect')

        with open(os.path.join(template_dir, 'in.defect'), 'r') as f:
            defect_template = f.read()

        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=self.alat)
        lattice_calculator = LatticeConstant(ff_settings=self.ff_settings)
        a, _, _ = lattice_calculator.calculate([unit_cell])[0]
        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice,
                                       alat=a)

        if self.lattice == 'fcc':
            idx, scale_factor = 95, [3, 3, 3]
        elif self.lattice == 'bcc':
            idx, scale_factor = 40, [3, 3, 3]
        elif self.lattice == 'diamond':
            idx, scale_factor = 7, [2, 2, 2]
        else:
            raise ValueError("Lattice type is invalid.")

        super_cell = unit_cell * scale_factor
        efs_calculator = EnergyForceStress(ff_settings=self.ff_settings)
        energy_per_atom = efs_calculator.calculate([super_cell])[0][0] / len(super_cell)

        super_cell_ld = LammpsData.from_structure(super_cell, atom_style='atomic')
        super_cell_ld.write_file('data.supercell')

        input_file = 'in.defect'

        with open(input_file, 'w') as f:
            f.write(defect_template.format(ff_settings='\n'.join(self.ff_settings.write_param()),
                                           lattice=self.lattice, alat=a, specie=self.specie,
                                           del_id=idx + 1, relaxed_file='data.relaxed'))

        return input_file, energy_per_atom, len(super_cell) - 1

    def calculate(self):
        """
        Calculate the vacancy formation given Potential class.
        """
        with ScratchDir('.'):
            input_file, energy_per_atom, num_atoms = self._setup()
            p = subprocess.Popen([self.LMP_EXE, '-in', input_file], stdout=subprocess.PIPE)
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
            defect_energy, _, _ = self._parse()
        defect_formation_energy = defect_energy - energy_per_atom * num_atoms

        return defect_formation_energy

    def _sanity_check(self, structure):
        return True

    def _parse(self):
        energy = float(np.loadtxt('energy.txt'))
        force = _read_dump('force.dump')
        stress = np.loadtxt('stress.txt')
        return energy, force, stress
