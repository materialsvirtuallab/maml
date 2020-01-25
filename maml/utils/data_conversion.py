import pandas as pd
import numpy as np
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
    energy = energy if energy else 0
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


def convert_docs(docs, **kwargs):
    """
    Method to convert a list of docs into objects, e.g.,
    Structure and DataFrame.

    Args:
        docs ([dict]): List of docs. Each doc should have the same
            format as one returned from .dft.parse_dir.

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
        y = np.concatenate(([outputs['energy']], force_arr))
        y_orig.append(y)
        n.append(np.insert(np.ones(len(y) - 1), 0, d['num_atoms']))
        dtype.extend(['energy'] + ['force'] * len(force_arr))
        structures.append(structure)
    df = pd.DataFrame(dict(y_orig=np.concatenate(y_orig), n=np.concatenate(n),
                           dtype=dtype))
    for k, v in kwargs.items():
        df[k] = v
    return structures, df
