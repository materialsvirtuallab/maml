# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""This module provides SNAP interatomic potential class."""

import re
import numpy as np
from monty.io import zopen
from sklearn.linear_model import LinearRegression

from ._base import Potential
from ._lammps import EnergyForceStress
from maml import ModelWithSklearn
from maml.describer import BispectrumCoefficients
from maml.utils import pool_from, convert_docs


class SNAPotential(Potential):
    """
    This class implements Spectral Neighbor Analysis Potential.
    """

    pair_style = 'pair_style        snap'
    pair_coeff = 'pair_coeff        * * {coeff_file} {param_file} {elements}'

    def __init__(self, model, name=None):
        """
        Initialize the SNAPotential Potential with atomic describer
        and model, which are used to generate the Bispectrum coefficients
        features for structures and to train the parameters.

        Args:
            model (LinearModel): Model to perform supervised learning with
                atomic descriptos as features and properties as targets.
            name (str): Name of force field.
        """
        self.name = name if name else 'SNAPotential'
        self.model = model
        self.elements = self.model.describer.elements

    def train(self, train_structures, train_energies, train_forces,
              train_stresses=None, **kwargs):
        """
        Training data with model.

        Args:
            train_structures ([Structure]): The list of Pymatgen Structure object.
                energies ([float]): The list of total energies of each structure
                in structures list.
            train_energies ([float]): List of total energies of each structure in
                structures list.
            train_forces ([np.array]): List of (m, 3) forces array of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            train_stresses (list): List of (6, ) virial stresses of each
                structure in structures list.
        """
        train_pool = pool_from(train_structures, train_energies, train_forces,
                               train_stresses)
        _, df = convert_docs(train_pool)
        ytrain = df['y_orig'] / df['n']
        xtrain = self.model.describer.transform(train_structures)
        self.model.fit(features=xtrain, targets=ytrain, **kwargs)

    def evaluate(self, test_structures, test_energies, test_forces,
                 test_stresses=None):
        """
        Evaluate energies, forces and stresses of structures with trained
        machinea learning potentials.

        Args:
            test_structures ([Structure]): List of Pymatgen Structure Objects.
            test_energies ([float]): List of DFT-calculated total energies of
                each structure in structures list.
            test_forces ([np.array]): List of DFT-calculated (m, 3) forces of
                each structure with m atoms in structures list. m can be varied
                with each single structure case.
            test_stresses (list): List of DFT-calculated (6, ) viriral stresses
                of each structure in structures list.
        """
        predict_pool = pool_from(test_structures, test_energies, test_forces,
                                 test_stresses)
        _, df_orig = convert_docs(predict_pool)
        _, df_predict = convert_docs(pool_from(test_structures))
        outputs = self.model.predict_objs(objs=test_structures)
        df_predict['y_orig'] = df_predict['n'] * outputs

        return df_orig, df_predict

    def predict_efs(self, structure):
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

    def write_param(self):
        """
        Write parameter and coefficient file to perform lammps calculation.
        """

        param_file = '{}.snapparam'.format(self.name)
        coeff_file = '{}.snapcoeff'.format(self.name)

        model = self.model
        describer = self.model.describer
        profile = describer.element_profile
        ne = len(self.elements)
        nbc = len(describer.subscripts)
        if describer.quadratic:
            nbc += int((1 + nbc) * nbc / 2)

        coeff_lines = []
        coeff_lines.append('{} {}'.format(ne, nbc + 1))
        for element, coeff in zip(self.elements, np.split(model.model.coef_, ne)):
            coeff_lines.append('{} {} {}'.format(element,
                                                 profile[element]['r'],
                                                 profile[element]['w']))
            coeff_lines.extend([str(c) for c in coeff])
        with open(coeff_file, 'w') as f:
            f.write('\n'.join(coeff_lines))

        param_lines = []
        keys = ['rcutfac', 'twojmax']
        param_lines.extend(['{} {}'.format(k, getattr(describer, k)) for k in keys])
        param_lines.extend(['rfac0 0.99363', 'rmin0 0'])
        param_lines.append('quadraticflag {}'.format(int(describer.quadratic)))
        param_lines.append('bzeroflag 0')
        with open(param_file, 'w') as f:
            f.write('\n'.join(param_lines))

        pair_style = self.pair_style
        pair_coeff = self.pair_coeff.format(elements=' '.join(self.elements),
                                            coeff_file=coeff_file,
                                            param_file=param_file)
        ff_settings = [pair_style, pair_coeff]
        return ff_settings

    def save(self, filename):
        """
        Save parameters of the potentials.

        Args:
            filename (str): The file to store parameters of potentials.

        Returns:
            (str)
        """
        self.model.save(model_fname=filename)
        return filename

    @staticmethod
    def from_config(param_file, coeff_file, **kwargs):
        """
        Initialize potentials with parameters file and coefficient file.

        Args:
            param_file (str): The file storing the configuration of potentials.
            coeff_file (str): The file storing the coefficients of potentials.

        Return:
            SNAPotential.
        """
        with open(coeff_file) as f:
            coeff_lines = f.readlines()
        coeff_lines = [line for line in coeff_lines if not line.startswith('#')]
        element_profile = {}
        ne, nbc = coeff_lines[0].split()
        ne, nbc = int(ne), int(nbc)
        for n in range(ne):
            specie, r, w = coeff_lines[1 + n * (nbc + 1)].split()
            r, w = float(r), float(w)
            element_profile[specie] = {'r': r, 'w': w}

        rcut_pattern = re.compile(r'rcutfac (.*?)\n', re.S)
        twojmax_pattern = re.compile(r'twojmax (\d*)\n', re.S)
        quadratic_pattern = re.compile(r'quadraticflag (.*?)(?=\n|$)', re.S)

        with zopen(param_file, 'rt') as f:
            param_lines = f.read()

        rcut = float(rcut_pattern.findall(param_lines)[-1])
        twojmax = int(twojmax_pattern.findall(param_lines)[-1])
        if quadratic_pattern.findall(param_lines):
            quadratic = bool(int(quadratic_pattern.findall(param_lines)[-1]))
        else:
            quadratic = False

        describer = BispectrumCoefficients(cutoff=rcut, twojmax=twojmax,
                                           element_profile=element_profile,
                                           quadratic=quadratic, pot_fit=True)
        model = ModelWithSklearn(model=LinearRegression(),
                                 describer=describer, **kwargs)
        coef = np.array(np.concatenate([coeff_lines[
                                        (2 + nbc * n + n): (2 + nbc * (n + 1) + n)]
                                        for n in range(ne)]), dtype=np.float)
        model.model.coef_ = coef
        model.model.intercept_ = 0
        snap = SNAPotential(model=model)
        return snap
