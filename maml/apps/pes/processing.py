# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import random
import numpy as np
import pandas as pd
from copy import copy
from pymatgen import Structure


def doc_from(structure, energy=None, force=None, stress=None):
    """
    Method to convert structure and its properties into doc
    format for further processing. If properties are None, zeros
    array will be used.
    Args:
        structure (Structure): Pymatgen Structure object.
        energy (float): The total energy of the structure.
        force (np.array): The (m, 3) forces array of the structure
            where m is the number of atoms in structure.
        stress (list/np.array): The (6, ) stresses array of the
            structure.
    Returns:
        (dict)
    """
    energy = energy if energy else 0.0
    force = force if force else np.zeros((len(structure), 3))
    stress = stress if stress else np.zeros(6)
    outputs = dict(energy=energy, forces=force,
                   virial_stress=stress)
    doc = dict(structure=structure.as_dict(),
               num_atoms=len(structure),
               outputs=outputs)
    return doc


def pool_from(structures, energies=None, forces=None, stresses=None):
    """
    Method to convert structures and their properties in to
    datapool format.
    Args:
        structures ([Structure]): The list of Pymatgen Structure object.
        energies ([float]): The list of total energies of each structure
            in structures list.
        forces ([np.array]): List of (m, 3) forces array of each structure
            with m atoms in structures list. m can be varied with each
            single structure case.
        stresses (list): List of (6, ) virial stresses of each
            structure in structures list.
    Returns:
        ([dict])
    """
    energies = energies if energies else [None] * len(structures)
    forces = forces if forces else [None] * len(structures)
    stresses = stresses if stresses else [None] * len(structures)
    datapool = [doc_from(structure, energy, force, stress)
                for structure, energy, force, stress
                in zip(structures, energies, forces, stresses)]
    return datapool


def convert_docs(docs, include_stress=False, **kwargs):
    """
    Method to convert a list of docs into objects, e.g.,
    Structure and DataFrame.
    Args:
        docs ([dict]): List of docs. Each doc should have the same
            format as one returned from .dft.parse_dir.
        include_stress (bool): Whether to include stress.
    Returns:
        A list of structures, and a DataFrame with energy and force
        data in 'y_orig' column, data type ('energy' or 'force') in
        'dtype' column, No. of atoms in 'n' column sharing the same row
        of energy data while 'n' being 1 for the rows of force data.
    """
    structures, y_orig, n, dtype = [], [], [], []
    for d in docs:
        if isinstance(d['structure'], dict):
            structure = Structure.from_dict(d['structure'])
        else:
            structure = d['structure']
        outputs = d['outputs']
        force_arr = np.array(outputs['forces'])
        assert force_arr.shape == (len(structure), 3), \
            'Wrong force array not matching structure'
        force_arr = force_arr.ravel()

        if include_stress:
            stress_arr = np.array(outputs['virial_stress'])
            y = np.concatenate(([outputs['energy']], force_arr, stress_arr))
            n.append(np.insert(np.ones(len(y) - 1), 0, d['num_atoms']))
            dtype.extend(['energy'] + ['force'] * len(force_arr) + ['stress'] * 6)
        else:
            y = np.concatenate(([outputs['energy']], force_arr))
            n.append(np.insert(np.ones(len(y) - 1), 0, d['num_atoms']))
            dtype.extend(['energy'] + ['force'] * len(force_arr))
        y_orig.append(y)
        structures.append(structure)
    df = pd.DataFrame(dict(y_orig=np.concatenate(y_orig), n=np.concatenate(n),
                           dtype=dtype))
    for k, v in kwargs.items():
        df[k] = v
    return structures, df


class MonteCarloSampler(object):
    """
    Sample a subset from the dataset to achieve some criteria.
    For example, one needs to subset the data so that a fraction
    of the data can already cover a large feature space,
    i.e., maximizing the distances.
    """

    def __init__(self, datasets, num_samples, cost_function):
        """
        Sample a subset with size num_samples from datasets
        to minimize the cost function.
        Args:
            datasets (numpy.array): The total datasets.
            num_samples (int): Number of samples from the data.
            cost_function (function): Function that takes into
                a subset of the data and calculate a cost.
        """
        self.datasets = datasets
        self.num_samples = num_samples
        self.cost_function = cost_function
        self.num_total = len(datasets)
        self.num_remain = self.num_total - num_samples
        self.index_selected = list(np.random.choice(
            self.num_total, num_samples, replace=False))
        self._get_remain_index()

        self.cost = self.compute_cost(self.datasets[self.index_selected, :])
        self.accepted = 0
        self.rejected = 0
        self.cost_history = []
        self.cost_history.append(self.cost)

    def _get_remain_index(self):
        self.index_remain = sorted(list(set(range(self.num_total)) -
                                        set(self.index_selected)))

    def compute_cost(self, data_subset):
        """
        Compute the cost of data subsets.
        Args:
            data_subset (numpy.array): Data subset.
        """
        return self.cost_function(data_subset)

    def sample(self, num_attempts, t_init, t_final):
        """
        Metropolis sampler. For every sampling attempt, one data entry is
        swapped with the data reservior. Then the energy difference is evaluated.
        If dE < 0, the swapping is accepted. If dE > 0, then it is accepted with
        probability exp(-dE / T), where T is some artificial temperature. We can
        start with a relatively large T, and then reduce it with sampling process
        going on.
        Args:
            num_attempts (int): Number of sampling attempts.
            t_init (float): Initial temperature.
            t_final (float): Final temperature.
        """
        temperatures = np.linspace(t_init, t_final, num_attempts)
        for i in range(num_attempts):
            temperature = temperatures[i]
            index = random.choice(self.index_selected)
            index_remain = random.choice(self.index_remain)
            self.update(index, index_remain, temperature)
            self.cost_history.append(self.cost)

    def update(self, index, index_remain, temperature):
        """
        Implement the data swap, if it is accepted.
        Args:
            index (int): The index of selected feature matrix
                used for swapping.
            index_remain (int): The index of remaining feature matrix
                used for swapping.
            temperature (float): Artificial temperature.
        """
        new_selected = copy(self.index_selected)
        new_selected.remove(index)
        new_selected.append(index_remain)

        cost_after_swap = self.compute_cost(self.datasets[new_selected, :])
        d_cost = cost_after_swap - self.cost
        accept = self.decision(d_cost, temperature)
        if accept:
            self.index_selected = copy(new_selected)
            self._get_remain_index()
            self.cost = cost_after_swap
        else:
            pass

    def decision(self, d_cost, temperature):
        """
        Decision on accepting the data swap.
        Args:
            d_cost (float): Difference between cost in proposed move.
            temperature (float): Temperature.
        """
        if d_cost < 0:
            self.accepted += 1
            return True
        else:
            p = np.exp(-d_cost / temperature)
            p2 = np.random.rand(1)
            if p2 < p:
                self.accepted += 1
                return True
            else:
                self.rejected += 1
                return False
