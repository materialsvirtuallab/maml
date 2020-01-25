# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import re
import os
import glob
import subprocess
import itertools
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from monty.io import zopen
from monty.os.path import which
from monty.tempfile import ScratchDir
from monty.serialization import loadfn
from pymatgen import Structure, Lattice, Element
from pymatgen.core import units
from .abstract import Potential
from .processing import pool_from, convert_docs
from .lammps.calcs import EnergyForceStress

module_dir = os.path.dirname(__file__)
NNinput_params = loadfn(os.path.join(module_dir, 'params', 'NNinput.json'))


class NNPotential(Potential):
    """
    This class implements Neural Network Potential.
    """
    bohr_to_angstrom = units.bohr_to_angstrom
    eV_to_Ha = units.eV_to_Ha
    pair_style = 'pair_style        nnp dir "./" showew no showewsum 0 ' \
                 'maxew 10000000 resetew yes cflength 1.8897261328 cfenergy 0.0367493254'
    pair_coeff = 'pair_coeff        * * {}'

    def __init__(self, name=None):
        """

        Args:
            name (str): Name of force field.
        """
        self.name = name if name else "NNPotential"
        self.specie = None
        self.weights = []
        self.bs = []
        self.atom_energy = None
        self.normalized_nodes = None
        self.epochs = None
        self.params = None
        self.scaling_params = None
        self.fitted = False

    def _line_up(self, structure, energy, forces, virial_stress):
        """
        Convert input structure, energy, forces, virial_stress to
        proper configuration format for RuNNer usage. Note that
        RuNNer takes bohr as length unit and Hatree as energy unit.

        Args:
            structure (Structure): Pymatgen Structure object.
            energy (float): DFT-calculated energy of the system.
            forces (list): The forces should have dimension
                (num_atoms, 3).
            virial_stress (list): stress should has 6 distinct
                elements arranged in order [xx, yy, zz, xy, yz, xz].

        Returns:
        """
        if len(structure.symbol_set) > 1:
            raise ValueError("Structure is not unary.")

        inputs = OrderedDict(Size=structure.num_sites,
                             SuperCell=structure.lattice,
                             AtomData=(structure, forces),
                             Energy=energy,
                             Stress=virial_stress)

        lines = ['begin']

        if 'SuperCell' in inputs:
            bohr_matrix = inputs['SuperCell'].matrix / self.bohr_to_angstrom
            for vec in bohr_matrix:
                lines.append('lattice {:>15.6f}{:>15.6f}{:>15.6f}'.format(*vec))
        if 'AtomData' in inputs:
            format_float = \
                'atom{:>16.9f}{:>16.9f}{:>16.9f}{:>4s}{:>15.9f}{:>15.9f}{:>15.9f}{:>15.9f}{:>15.9f}'
            for i, (site, force) in enumerate(zip(structure, forces)):
                lines.append(format_float.format(*site.coords / self.bohr_to_angstrom,
                                                 site.species_string, 0.0, 0.0,
                                                 *np.array(force) * self.eV_to_Ha * self.bohr_to_angstrom))
        if 'Energy' in inputs:
            lines.append('energy  {:f}'.format(energy * self.eV_to_Ha))

        lines.append('charge  {:f}'.format(structure.charge))
        lines.append('end')

        return '\n'.join(lines)

    def write_cfgs(self, filename, cfg_pool):

        lines = []
        for dataset in cfg_pool:
            if isinstance(dataset['structure'], dict):
                structure = Structure.from_dict(dataset['structure'])
            else:
                structure = dataset['structure']
            energy = dataset['outputs']['energy']
            forces = dataset['outputs']['forces']
            virial_stress = dataset['outputs']['virial_stress']

            lines.append(self._line_up(structure, energy, forces, virial_stress))

            # dist = np.unique(structure.distance_matrix.ravel())[1]
            # if self.shortest_distance > dist:
            #     self.shortest_distance = dist

        self.specie = Element(structure.symbol_set[0])

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        return filename

    def write_input(self, **kwargs):
        """
        Write input.nn file to train the Neural Network model.

        Args:
            atom_energy (float): Atomic reference energy.

            kwargs:
                General nnp settings:
                    atom_energy (None): Free atom reference energy.
                    cutoff_type (int): Type of cutoff function. Default to 1
                        (i.e., cosine function).
                    scale_features (int): Determine the method to scale the
                        symmetry function.
                        0: no scaling.
                        1: scale_symmetry_functions.
                        2: center_symmetry_functions.
                        3. scale_symmetry_functions_sigma.
                    scale_min_short (float): Minimum value for scaling.
                        Default to 0.0.
                    scale_max_short (float): Maximum value for scaling.
                        Default to 1.
                    hidden_layers (list): List of the numbers of
                        nodes in each hidden layer.
                    activations (str): Activation function for each hidden layer.
                        't': tanh, 's': logistic, 'p': softplus.
                    normalize_nodes (boolean): Whether to normalize input of nodes.

                Additional settings for training:
                    epoch (int): Number of training epochs.
                    updater_type (int): Weight update method
                        0: gradient Descent, 1: Kalman filter.
                    parallel_mode (int): Training parallelization used.
                        Default to serial mode.
                    update_strategy (int): Update strategy.
                        0: combined, 1: per-element.
                    selection_mode (int): Update candidate selection mode.
                        0: random, 1: sort, 2: threshold
                    test_fraction (float): Fraction of structures kept for
                        testing.
                    force_weight (float): Weight of force updates relative
                        to energy updates. Default to 10.0
                    short_energy_fraction (float): Fraction of energy updates
                        per epoch. Default to 1.0.
                    short_force_fraction (float): Fraction of force updates
                        per epoch. Default to 0.02315.
                    short_energy_error_threshold (float): RMSE threshold for
                        energy update candidates. Default to 0.0.
                    short_force_error_threshold (float): RMSE threshold for
                        force update candidates. Default to 1.0.
                    rmse_threshold_trials (int): Maximum number of RMSE
                        threshold trials. Default to 3.
                    weights_min (float): Minimum value for initial random
                        weights. Default to -1.
                    weights_max (float): Maximum value for initial random
                        weights. Default to 1.
                    write_trainpoints (int): Write energy comparison every
                        this many epochs. Default to 1.
                    write_trainforces (int): Write force comparison every
                        this many epochs. Default to 1.
                    write_weights_epoch (int): Write weights every this many
                        epochs. Default to 1.
                    write_neuronstats (int): Write neuron statistics every
                        this many epochs. Default to 1.

                    # Kalman Filter
                    kalman_type (int): Kalman filter type. Default to 0.
                    kalman_epsilon (float): General Kalman filter parameter
                        epsilon. Default to 0.01.
                    kalman_q0 (float): General Kalman filter parameter q0.
                        Default to 0.01.
                    kalman_qtau (float): General Kalman filter parameter
                        qtau. Default to 2.302.
                    kalman_qmin (float): General Kalman filter parameter qmin.
                        Default to 1e-6.
                    kalman_eta (float): Standard Kalman filter parameter eta.
                        Default to 0.01.
                    kalman_etatau (float): Standard Kalman filter parameter
                        etatau. Defaul to 2.302.
                    kalman_etamax (float): Standard Kalman filter parameter
                        etamax. Default to 1.0.

                Symmetry functions:
                    r_cut (float): Cutoff distance (unit: Å).
                    r_etas (numpy.array): η in radial function.
                    r_shift (numpy.array): Rs in radial function.
                    a_etas (numpy.array): η in angular function.
                    zetas (numpy.array): ζ in angular function.
                    lambdas (numpy.array): λ in angular function. Default to (1, -1).
        """
        filename = 'input.nn'

        head_formatter = '{:<32s}{value}'
        type2_format = 'symfunction_short {central_atom}  2 {neighbor_atom}' \
                       '    {r_eta:.7f}    {rs:.7f}    {rcut:.7f}'
        type3_format = 'symfunction_short {central_atom}  3 {neighbor_atom1} ' \
                       '{neighbor_atom2}    {a_eta:.7f} {lambd:>2d} {zeta:.7f}   ' \
                       '{rcut:.7f}'

        specie = self.specie.name
        lines = [head_formatter.format('number_of_elements', value=1),
                 head_formatter.format('elements', value=specie)]

        PARAMS = {'general': ['cutoff_type', 'scale_features', 'scale_min_short',
                              'scale_max_short', 'hidden_layers'],
                  'additional': ['epochs', 'updater_type', 'parallel_mode',
                                 'update_strategy', 'selection_mode', 'random_seed',
                                 'test_fraction', 'force_weight', 'short_energy_fraction',
                                 'short_force_fraction', 'short_energy_error_threshold',
                                 'short_force_error_threshold', 'rmse_threshold_trials',
                                 'weights_min', 'weights_max', 'write_trainpoints',
                                 'write_trainforces', 'write_weights_epoch',
                                 'write_neuronstats', 'kalman_type', 'kalman_epsilon',
                                 'kalman_q0', 'kalman_qtau', 'kalman_qmin', 'kalman_eta',
                                 'kalman_etatau', 'kalman_etamax']}
        if self.fitted:
            if self.atom_energy:
                lines.append(head_formatter.format('atom_energy',
                                                   value=' '.join([specie, str(self.atom_energy)])))
            for tag in PARAMS.get('general'):
                if tag == 'scale_features':
                    lines.append(NNinput_params.get('general').get(tag).get(getattr(self, tag)))
                elif tag == 'hidden_layers':
                    layers = self.hidden_layers
                    activations = self.activations
                    lines.append(head_formatter.format('global_hidden_layers_short',
                                                       value=len(layers)))
                    lines.append(head_formatter.format('global_nodes_short',
                                                       value=' '.join([str(i) for i in layers])))
                    lines.append(head_formatter.format('global_activation_short',
                                                       value=' '.join([activations] * len(layers) + ['l'])))
                else:
                    lines.append(head_formatter.format(tag, value=getattr(self, tag)))
            if self.normalized_nodes:
                lines.append('normalize_nodes')

            for tag in PARAMS.get('additional'):
                lines.append(head_formatter.format(tag, value=getattr(self, tag)))
            lines.append('use_short_forces')

            central_atom, neighbor_atom1, neighbor_atom2 = specie, specie, specie

            r_cut = self.r_cut
            r_cut /= self.bohr_to_angstrom
            r_shift = np.array(self.r_shift)
            r_shift /= self.bohr_to_angstrom

            for r_eta, rs in itertools.product(self.r_etas, r_shift):
                lines.append(type2_format.format(central_atom=central_atom,
                                                 neighbor_atom=neighbor_atom1,
                                                 r_eta=r_eta, rs=rs, rcut=r_cut))

            for a_eta, lambd, zeta in itertools.product(self.a_etas, self.lambdas,
                                                        self.zetas):
                lines.append(type3_format.format(central_atom=central_atom,
                                                 neighbor_atom1=neighbor_atom1,
                                                 neighbor_atom2=neighbor_atom2,
                                                 a_eta=a_eta, lambd=lambd,
                                                 zeta=zeta, rcut=r_cut))
        else:
            if kwargs.get('atom_energy'):
                lines.append(head_formatter.format('atom_energy',
                                                   value=' '.join([specie, str(kwargs.get('atom_energy'))])))
                setattr(self, 'atom_energy', kwargs.get('atom_energy'))
            for tag in PARAMS.get('general'):
                if tag == 'scale_features':
                    value = kwargs.get(tag) if kwargs.get(tag) is not None else '1'
                    lines.append(NNinput_params.get('general').get(tag).get(value))
                    setattr(self, tag, value)
                elif tag == 'hidden_layers':
                    layers = kwargs.get(tag) if kwargs.get(tag) is not None \
                        else NNinput_params.get('general').get(tag)
                    setattr(self, tag, layers)
                    activations = kwargs.get('activations') if kwargs.get('activations') is not None \
                        else NNinput_params.get('general').get('activations')
                    setattr(self, 'activations', activations)
                    lines.append(head_formatter.format('global_hidden_layers_short',
                                                       value=len(layers)))
                    lines.append(head_formatter.format('global_nodes_short',
                                                       value=' '.join([str(i) for i in layers])))
                    lines.append(head_formatter.format('global_activation_short',
                                                       value=' '.join([activations] * len(layers) + ['l'])))
                else:
                    value = kwargs.get(tag) if kwargs.get(tag) is not None \
                        else NNinput_params.get('general').get(tag)
                    lines.append(head_formatter.format(tag, value=value))
                    setattr(self, tag, value)
            if kwargs.get('normalize_nodes'):
                lines.append('normalize_nodes')
                setattr(self, 'normalize_nodes', True)

            for tag in PARAMS.get('additional'):
                value = kwargs.get(tag) if kwargs.get(tag) is not None \
                    else NNinput_params.get('additional').get(tag)
                lines.append(head_formatter.format(tag, value=value))
                setattr(self, tag, value)
            lines.append('use_short_forces')

            central_atom, neighbor_atom1, neighbor_atom2 = specie, specie, specie

            r_cut = kwargs.get('r_cut') if kwargs.get('r_cut') is not None \
                else NNinput_params.get('symmetry_function').get('r_cut')
            setattr(self, 'r_cut', r_cut)
            r_cut /= self.bohr_to_angstrom
            r_etas = kwargs.get('r_etas') if kwargs.get('r_etas') is not None \
                else NNinput_params.get('symmetry_function').get('r_etas')
            setattr(self, 'r_etas', r_etas)
            r_shift = kwargs.get('r_shift') if kwargs.get('r_shift') is not None \
                else NNinput_params.get('symmetry_function').get('r_shift')
            setattr(self, 'r_shift', r_shift)
            r_shift = np.array(r_shift)
            r_shift /= self.bohr_to_angstrom
            a_etas = kwargs.get('a_etas') if kwargs.get('a_etas') is not None \
                else NNinput_params.get('symmetry_function').get('a_etas')
            setattr(self, 'a_etas', a_etas)
            zetas = kwargs.get('zetas') if kwargs.get('zetas') is not None \
                else NNinput_params.get('symmetry_function').get('zetas')
            setattr(self, 'zetas', zetas)
            lambdas = kwargs.get('lambdas') if kwargs.get('lambdas') is not None \
                else NNinput_params.get('symmetry_function').get('lambdas')
            setattr(self, 'lambdas', lambdas)

            for r_eta, rs in itertools.product(r_etas, r_shift):
                lines.append(type2_format.format(central_atom=central_atom,
                                                 neighbor_atom=neighbor_atom1,
                                                 r_eta=r_eta, rs=rs, rcut=r_cut))

            for a_eta, lambd, zeta in itertools.product(a_etas, lambdas, zetas):
                lines.append(type3_format.format(central_atom=central_atom,
                                                 neighbor_atom1=neighbor_atom1,
                                                 neighbor_atom2=neighbor_atom2,
                                                 a_eta=a_eta, lambd=lambd,
                                                 zeta=zeta, rcut=r_cut))

            self.num_symm_functions = len(list(itertools.product(r_etas, r_shift))) + \
                len(list(itertools.product(a_etas, lambdas, zetas)))
            self.layer_sizes = [self.num_symm_functions] + self.hidden_layers

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        self.fitted = True

        return filename

    def read_cfgs(self, filename='output.data'):
        """
        Args:
            filename (str): The configuration file to be read.
        """
        data_pool = []
        with zopen(filename, 'rt') as f:
            lines = f.read()

        block_pattern = re.compile('begin\n(.*?)end', re.S)
        lattice_pattern = re.compile('lattice(.*?)\n')
        position_pattern = re.compile('atom(.*?)\n')
        energy_pattern = re.compile('energy(.*?)\n')

        for block in block_pattern.findall(lines):
            d = {'outputs': {}}
            lattice_str = lattice_pattern.findall(block)
            lattice = Lattice(np.array([latt.split() for latt in lattice_str],
                                       dtype=np.float) * self.bohr_to_angstrom)
            position_str = position_pattern.findall(block)
            positions = pd.DataFrame([pos.split() for pos in position_str])
            positions.columns = \
                ['x', 'y', 'z', 'specie', 'charge', 'atomic_energy', 'fx', 'fy', 'fz']
            coords = np.array(positions.loc[:, ['x', 'y', 'z']], dtype=np.float)
            coords = coords * self.bohr_to_angstrom
            species = np.array(positions['specie'])
            forces = np.array(positions.loc[:, ['fx', 'fy', 'fz']], dtype=np.float)
            forces = forces / self.eV_to_Ha / self.bohr_to_angstrom
            energy_str = energy_pattern.findall(block)[0]
            energy = float(energy_str.lstrip()) / self.eV_to_Ha
            struct = Structure(lattice=lattice, species=species, coords=coords,
                               coords_are_cartesian=True)
            d['structure'] = struct.as_dict()
            d['outputs']['energy'] = energy
            d['outputs']['forces'] = forces
            d['num_atoms'] = len(struct)

            data_pool.append(d)
        _, df = convert_docs(docs=data_pool)
        return data_pool, df

    def write_param(self):
        """
        Write optimized weights file to perform energy and force prediction.
        """
        if self.params is None or self.scaling_params is None:
            raise RuntimeError("The parameters should be provided.")
        weights_filename = '.'.join(['weights', self.suffix, 'data'])
        weight_formatter = '{:>18s}{:>2s}{:>10s}{:>6s}{:>6s}{:>6s}{:>6s}'
        bias_formatter = '{:>18s}{:>2s}{:>10s}{:>6s}{:>6}'
        lines = []
        for i in range(self.params.shape[0]):
            if self.params.iloc[i]['type'] == 'a':
                lines.append(weight_formatter.format(*self.params.iloc[i]))
            else:
                lines.append(bias_formatter.format(*self.params.iloc[i]))

        with open(weights_filename, 'w') as f:
            f.writelines('\n'.join(lines))

        scaling_filename = 'scaling.data'
        scaling_formatter = '{:>4s}{:>5s}  {:>22s} {:>22s} {:>22s} {:.>22s}'
        scaling_lines = []
        for i in range(self.num_symm_functions):
            scaling_lines.append(scaling_formatter.format(*self.scaling_params.iloc[i]))
        with open(scaling_filename, 'w') as f:
            f.writelines('\n'.join(scaling_lines))

        self.write_input()

        ff_settings = [self.pair_style, self.pair_coeff.format(self.r_cut + 1e-2)]

        return ff_settings

    def train(self, train_structures, energies=None, forces=None, stresses=None,
              **kwargs):
        """
        Training data with moment tensor method.

        Args:
            train_structures ([Structure]): The list of Pymatgen Structure object.
                energies ([float]): The list of total energies of each structure
                in structures list.
            energies ([float]): List of total energies of each structure in
                structures list.
            forces ([np.array]): List of (m, 3) forces array of each structure
                with m atoms in structures list. m can be varied with each
                single structure case.
            stresses (list): List of (6, ) virial stresses of each
                structure in structures list.
            kwargs: Parameters in write_input method.
        """
        if not which('nnp-train'):
            raise RuntimeError("NNP Trainer has not been found.")

        train_pool = pool_from(train_structures, energies, forces, stresses)
        atoms_filename = 'input.data'

        with ScratchDir('.'):
            atoms_filename = self.write_cfgs(filename=atoms_filename, cfg_pool=train_pool)
            output = 'training_output'

            input_filename = self.write_input(**kwargs)
            p_scaling = subprocess.Popen(['nnp-scaling', input_filename])
            stdout = p_scaling.communicate()[0]

            p_train = subprocess.Popen(['nnp-train', input_filename],
                                       stdout=open(output, 'w'))
            stdout = p_train.communicate()[0]

            rc = p_train.returncode
            if rc != 0:
                error_msg = 'RuNNer exited with return code %d' % rc
                msg = stdout.decode("utf-8").split('\n')[:-1]
                try:
                    error_line = [i for i, m in enumerate(msg)
                                  if m.startswith('ERROR')][0]
                    error_msg += ', '.join([e for e in msg[error_line:]])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            with zopen(output) as f:
                error_lines = f.read()

            energy_rmse_pattern = re.compile(r'ENERGY\s*\S*\s*(\S*)\s*(\S*).*?\n')
            forces_rmse_pattern = re.compile(r'FORCES\s*\S*\s*(\S*)\s*(\S*).*?\n')
            self.train_energy_rmse, self.validation_energy_rmse = \
                np.array([line for line in energy_rmse_pattern.findall(error_lines)],
                         dtype=np.float).T
            self.train_forces_rmse, self.validation_forces_rmse = \
                np.array([line for line in forces_rmse_pattern.findall(error_lines)],
                         dtype=np.float).T

            weights_filename_pattern = 'weights*{}.out'.format(self.epochs)
            weights_filename = glob.glob(weights_filename_pattern)[0]

            self.suffix = weights_filename.split('.')[1]

            with open(weights_filename) as f:
                weights_lines = f.readlines()

            params = pd.DataFrame([line.split() for line in weights_lines
                                   if "#" not in line])
            params.columns = ['value', 'type', 'index', 'start_layer',
                              'start_neuron', 'end_layer', 'end_neuron']
            self.params = params

            for layer_index in range(1, len(self.layer_sizes)):
                weights_group = params[(params['start_layer'] == str(layer_index - 1))
                                       & (params['end_layer'] == str(layer_index))]

                weights = np.reshape(np.array(weights_group['value'], dtype=np.float),
                                     (self.layer_sizes[layer_index - 1],
                                      self.layer_sizes[layer_index]))
                self.weights.append(weights)

                bs_group = params[(params['type'] == 'b') &
                                  (params['start_layer'] == str(layer_index))]
                bs = np.array(bs_group['value'], dtype=np.float)
                self.bs.append(bs)

            with open('scaling.data') as f:
                scaling_lines = f.readlines()
            scaling_params = pd.DataFrame([line.split() for line in scaling_lines
                                           if '#' not in line])
            scaling_params.column = ['e_index', 'sf_index', 'sf_min', 'sf_max',
                                     'sf_mean', 'sf_sigma']
            self.scaling_params = scaling_params

        return rc

    def evaluate(self, test_structures, ref_energies, ref_forces, ref_stresses):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potential.

        Args:
            test_structures ([Structure]): List of Pymatgen Structure Objects.
            ref_energies ([float]): List of DFT-calculated total energies of
                each structure in structures list.
            ref_forces ([np.array]): List of DFT-calculated (m, 3) forces of
                each structure with m atoms in structures list. m can be varied
                with each single structure case.
            ref_stresses (list): List of DFT-calculated (6, ) viriral stresses
                of each structure in structures list.
        """
        if not which('nnp-predict'):
            raise RuntimeError("NNP Predictor has not been found.")

        original_file = 'input.data'
        predict_file = 'output.data'

        predict_pool = pool_from(test_structures, ref_energies,
                                 ref_forces, ref_stresses)
        with ScratchDir('.'):
            _, _ = self.write_param()
            original_file = self.write_cfgs(original_file, cfg_pool=predict_pool)
            _, df_orig = self.read_cfgs(original_file)

            input_filename = self.write_input()

            dfs = []
            for data in predict_pool:
                _ = self.write_cfgs(original_file, cfg_pool=[data])
                p = subprocess.Popen(['nnp-predict', input_filename], stdout=subprocess.PIPE)
                stdout = p.communicate()[0]

                rc = p.returncode
                if rc != 0:
                    error_msg = 'RuNNer exited with return code %d' % rc
                    msg = stdout.decode("utf-8").split('\n')[:-1]
                    try:
                        error_line = [i for i, m in enumerate(msg)
                                      if m.startswith('ERROR')][0]
                        error_msg += ', '.join([e for e in msg[error_line:]])
                    except Exception:
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)

                _, df = self.read_cfgs(predict_file)
                dfs.append(df)
            df_predict = pd.concat(dfs, ignore_index=True)

        return df_orig, df_predict

    def predict(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """
        calculator = EnergyForceStress(self)
        energy, forces, stress = calculator.calculate(structures=[structure])[0]
        return energy, forces, stress
