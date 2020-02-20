# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""This package contains Potential classes representing Interatomic Potentials."""

import abc
import six
from monty.json import MSONable


class Potential(six.with_metaclass(abc.ABCMeta, MSONable)):
    """
    Abstract Base class for a Interatomic Potential.
    """

    @abc.abstractmethod
    def train(self, train_structures, energies, forces, stresses, **kwargs):
        """
        Train interatomic potentials with energies, forces and
        stresses corresponding to structures.

        Args:
            train_structures (list): List of Pymatgen Structure objects.
            energies (list): List of DFT-calculated total energies of each structure
                in structures list.
            forces (list): List of DFT-calculated (m, 3) forces of each structure
                with m atoms in structures list. m can be varied with each single
                structure case.
            stresses (list): List of DFT-calculated (6, ) virial stresses of each
                structure in structures list.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, test_structures, ref_energies, ref_forces, ref_stresses):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures (list): List of Pymatgen Structure Objects.
            ref_energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            ref_forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            ref_stresses (list): List of DFT-calculated (6, ) viriral stresses of
                each structure in structures list.

        Returns:
            DataFrame of original data and DataFrame of predicted data.
        """
        pass

    @abc.abstractmethod
    def predict(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """
        pass
