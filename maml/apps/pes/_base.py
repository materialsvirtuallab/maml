# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
"""
Base classes for potentials
"""

import abc

from maml.base import BaseModel


class PotentialMixin:
    """
    Abstract base class for a Interatomic Potential.
    """

    @abc.abstractmethod
    def train(self, train_structures, train_energies, train_forces, train_stresses, **kwargs):
        """
        Train interatomic potentials with energies, forces ann stresses corresponding
            to structures.

        Args:
            train_structures (list): List of Pymatgen Structure objects.
            train_energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            train_forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            train_stresses (list): List of DFT-calculated (6, ) virial stresses of
                each structure in structures list.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, test_structures, test_energies, test_forces, test_stresses):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures (list): List of Pymatgen Structure Objects.
            test_energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            test_forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            test_stresses (list): List of DFT-calculated (6, ) viriral stresses of
                each structure in structures list.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_efs(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """

    @abc.abstractmethod
    def write_param(self):
        """
        Write optimized weights file to perform energy and force prediction via
        lammps calculation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_config(self, **kwargs):
        """
        Initialize potentials with parameters file.
        """
        raise NotImplementedError


class Potential(PotentialMixin, BaseModel):  # type: ignore
    """
    Potential models that can be used to fit structure-[energy, force, stress]
    pairs
    """
