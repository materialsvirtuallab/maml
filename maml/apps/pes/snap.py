# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import numpy as np
import pandas as pd
from pymatgen import Element
from .abstract import Potential
from .processing import pool_from, convert_docs
from .lammps.calcs import EnergyForceStress


class SNAPotential(Potential):
    """
    This class implements Spectral Neighbor Analysis Potential.
    """
    pair_style = 'pair_style        snap'
    pair_coeff = 'pair_coeff        * * {coeff_file} {elements} {param_file} {specie}'

    def __init__(self, model, name=None):
        """
        Initialize the SNAPotential Potential with atomic describer
        and model, which are used to generate the Bispectrum coefficients
        features for structures and to train the parameters.

        Args:
            model (Model): Model to perform supervised learning with
                atomic descriptos as features and properties as targets.
            name (str): Name of force field.
        """
        self.name = name if name else 'SNAPotential'
        self.model = model
        self.specie = None

    def train(self, train_structures, energies, forces, stresses=None, **kwargs):
        """
        Training data with model.

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
        """
        train_pool = pool_from(train_structures, energies, forces, stresses)
        _, df = convert_docs(train_pool)
        ytrain = df['y_orig'] / df['n']
        self.model.fit(inputs=train_structures, outputs=ytrain, **kwargs)
        self.specie = Element(train_structures[0].symbol_set[0])

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
        predict_pool = pool_from(test_structures, ref_energies,
                                 ref_forces, ref_stresses)
        _, df_orig = convert_docs(predict_pool)

        _, df_predict = convert_docs(pool_from(test_structures))
        outputs = self.model.predict(inputs=test_structures, override=True)
        df_predict['y_orig'] = df_predict['n'] * outputs

        return df_orig, df_predict

    def predict(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """
        # outputs = self.model.predict([structure])
        # energy = outputs[0]
        # forces = outputs[1:].reshape(len(structure), 3)
        calculator = EnergyForceStress(ff_settings=self)
        energy, forces, stress = calculator.calculate(structures=[structure])[0]
        return energy, forces, stress

    def write_param(self):
        """
        Write parameter and coefficient file to perform lammps calculation.
        """
        if not self.specie:
            raise ValueError("No specie given!")

        param_file = '{}.snapparam'.format(self.name)
        coeff_file = '{}.snapcoeff'.format(self.name)

        model = self.model
        # ncoeff = len(model.coef)
        describer = self.model.describer
        profile = describer.element_profile
        elements = [element.symbol for element
                    in sorted([Element(e) for e in profile.keys()])]
        ne = len(elements)
        nbc = len(describer.subscripts)
        if describer.quadratic:
            nbc += int((1 + nbc) * nbc / 2)
        tjm = describer.twojmax
        diag = describer.diagonalstyle
        # assert ncoeff == ne * (nbc + 1),\
        #     '{} coefficients given. '.format(ncoeff) + \
        #     '{} ({} * ({} + 1)) '.format(ne * (nbc + 1), ne, nbc) + \
        #     'coefficients expected ' + \
        #     'for twojmax={} and diagonalstyle={}.'.format(tjm, diag)

        coeff_lines = []
        coeff_lines.append('{} {}'.format(ne, nbc + 1))
        for element, coeff in zip(elements, np.split(model.coef, ne)):
            coeff_lines.append('{} {} {}'.format(element,
                                                 profile[element]['r'],
                                                 profile[element]['w']))
            coeff_lines.extend([str(c) for c in coeff])
        with open(coeff_file, 'w') as f:
            f.write('\n'.join(coeff_lines))

        param_lines = []
        keys = ['rcutfac', 'twojmax', 'rfac0', 'rmin0', 'diagonalstyle']
        param_lines.extend(['{} {}'.format(k, getattr(describer, k))
                            for k in keys])
        param_lines.append('quadraticflag {}'.format(int(describer.quadratic)))
        param_lines.append('bzeroflag 0')
        with open(param_file, 'w') as f:
            f.write('\n'.join(param_lines))

        pair_coeff = self.pair_coeff.format(elements=' '.join(elements),
                                            specie=self.specie.name,
                                            coeff_file=coeff_file,
                                            param_file=param_file)
        ff_settings = [self.pair_style, pair_coeff]
        return ff_settings

    def save(self, filename):
        """
        Save parameters of the potential.

        Args:
            filename (str): The file to store parameters of potential.

        Returns:
            (str)
        """
        self.model.save(filename=filename)
        return filename
