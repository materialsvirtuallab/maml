"""
Machine learning potential abstract classs
"""
# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import abc
from monty.json import MSONable

from maml import Model


class Potential(MSONable):
    """
    Abstract Base class for an interatomic Potential.
    """

    @abc.abstractmethod
    def train(self, train_structures, energies, forces, stresses, **kwargs):
        """
        Train interatomic potential with energies, forces and
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
        interatomic potential.

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


class Potentialmaml(Model):
    """
    Abstract Base class for a interatomic Potential.
    """

    def predict_structures(self, structures):
        features = self.describer.transform(structures)
        return self.predict(features)

    def train(self, train_structures, targets, **kwargs):
        """
        Train interatomic potential with energies, forces and
        stresses corresponding to structures.

        Args:
            train_structures (list): List of Pymatgen Structure objects.
            targets (list): targets that can be directly trained
        """
        features = self.describer.transform(train_structures)
        self.fit(features, targets, **kwargs)

    def evaluate_structures(self, test_structures, targets, metrics=['mae']):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potential.

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
        features = self.describer.transform(test_structures)
        evaluation = self.evaluate(features, targets, metrics=metrics, multi_targets=False)
        return evaluation
