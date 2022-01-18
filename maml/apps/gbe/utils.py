import os
from typing import Dict, List, Optional
from monty.serialization import dumpfn, loadfn
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

import maml.apps.gbe as gbe

pjoin = os.path.join
module_dir = os.path.dirname(gbe.__file__)
REFS = pjoin(module_dir, "references")
DATA = pjoin(module_dir, "data")


def load_data(filename: Optional[str] = None) -> List:
    """
    Helper function to load the data
    Default is to load the 361 data 
    Args:
        filename (str): the filename of the data

    Returns:
        data list

    """
    if filename:
        data = loadfn(filename)
    else:
        data = loadfn(pjoin(DATA, "gb_data.json"))
    return data


def load_b0_dict() -> Dict:
    """
    Helper function to retrieve the b0 data.
    b0 is the bulk bond length.
    Returns: dict
        returns the dict of {element: b0}
    """
    if os.path.isfile(pjoin(REFS, "el2b0.json")):
        return loadfn(pjoin(REFS, "el2b0.json"))
    print("Downloading Data ... ")
    data = load_data()
    b0_dict = update_b0_dict(data)
    dumpfn(b0_dict, pjoin(REFS, "el2b0.json"))


def update_b0_dict(data_source: list, mp_api: str) -> Dict:
    """
    Helper function to update the b0 dictionary
    Requires api key to MP
    Args:
        data_source (list): list of GB data
        mp_api (str): API key to MP
    Returns:
         b0_dict (dict): {el: b0}
    """
    rester = MPRester(mp_api)

    def get_b0(bulk_structure: Structure) -> float:
        """b0 is the bond length of bulk metal"""
        _b0 = bulk_structure.get_distance(0, 0, jimage=1)
        return min(_b0, bulk_structure.lattice.a)

    b0_dict = {}
    for d in data_source:
        if d['pretty_formula'] in b0_dict:
            continue
        bs = rester.get_structure_by_material_id(d['material_id'])
        b0 = get_b0(bs)
        b0_dict.update({d['pretty_formula']: b0})

    return b0_dict


def load_mean_delta_bl_dict(loc_algo: str = 'crystalnn') -> Dict:
    """
    Helper function to load the mean_delta_bl data
    Args:
        loc_algo (str): name of the algorithm

    Returns: dict
        {task_id (int): mean_delta_bl(float)}

    """
    if os.path.isfile(pjoin(REFS, "mean_bl_chg_{}.json".format(loc_algo))):
        return loadfn(pjoin(REFS, "mean_bl_chg_{}.json".format(loc_algo)))
