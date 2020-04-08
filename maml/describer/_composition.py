"""
Compositional describers
"""
from functools import partial
import os
from typing import Dict, List, Union, Optional

from matminer.featurizers.composition import ElementProperty as MatminerElementProperty  # noqa
from ._matminer_wrapper import wrap_matminer_describer
import numpy as np
from pymatgen.core import Composition, Structure, Element, Molecule, Specie
import pandas as pd
from sklearn.decomposition import PCA
import json

from maml.base import BaseDescriber, OutDataFrameConcat
from maml.utils import Stats, STATS_KWARGS, stats_list_conversion


CWD = os.path.abspath(os.path.dirname(__file__))

DATA_MAPPING = {'megnet_1': 'data/elemental_embedding_1MEGNet_layer.json',
                'megnet_3': 'data/elemental_embedding_3MEGNet_layer.json'}

for length in [2, 3, 4, 8, 16, 32]:
    DATA_MAPPING['megnet_l%d' % length] = 'data/elemental_embedding_1MEGNet_layer_length_%d.json' % length
    DATA_MAPPING['megnet_ion_l%d' % length] = 'data/ion_embedding_1MEGNet_layer_length_%d.json' % length

ElementProperty = wrap_matminer_describer("ElementProperty", MatminerElementProperty)


class ElementStats(OutDataFrameConcat, BaseDescriber):
    """
    Element statistics. The allowed stats are accessed via ALLOWED_STATS class
    attributes. If the stats have multiple parameters, the positional arguments
    are separated by ::, e.g., moment::1::None
    """

    ALLOWED_STATS = Stats.allowed_stats  # type: ignore
    AVAILABLE_DATA = list(DATA_MAPPING.keys())

    def __init__(self, element_properties: Dict, stats: List[str],
                 property_names: Optional[List[str]] = None, **kwargs):
        """
        Elemental stats for composition/str/structure

        Args:
            element_properties (dict): element properties, e.g.,
                {'H': [0.1, 0.2, 0.3], 'He': [0.12, ...]}
            stats (list): list of stats, check ElementStats.ALLOWED_STATS
                for supported stats. The stats that support additional
                Keyword args, use ':' to separate the args. For example,
                'moment:0:None' will calculate moment stats with order=0,
                and max_order=None.
            property_names (list): list of property names, has to be consistent
                in length with properties in element_properties
            **kwargs:
        """

        self.element_properties = element_properties
        properties = list(self.element_properties.values())

        n_property = list(set([len(i) for i in properties]))
        if len(n_property) > 1:
            raise ValueError("Property length not consistent")
        n_single_property = n_property[0]

        if property_names is None:
            property_names = ['p%d' % i for i in range(n_single_property)]

        if len(property_names) != n_single_property:
            raise ValueError("Property name length is not consistent")
        all_property_names = []

        stats_func = []
        full_stats = stats_list_conversion(stats)

        for stat in full_stats:
            all_property_names.extend(['%s_%s' % (p, stat) for p in property_names])
            if ':' in stat:
                splits = stat.split(":")
                stat_name = splits[0]

                if stat_name.lower() not in self.ALLOWED_STATS:
                    raise ValueError(f"{stat_name.lower()} not in available Stats")

                func = getattr(Stats, stat_name)
                args = splits[1:]
                arg_dict = {}
                for name_dict, arg in zip(STATS_KWARGS[stat_name], args):
                    name = list(name_dict.keys())[0]
                    value_type = list(name_dict.values())[0]
                    try:
                        value = value_type(arg)
                    except ValueError:
                        value = None  # type: ignore
                    arg_dict[name] = value
                stats_func.append(partial(func, **arg_dict))
                continue

            if stat.lower() not in self.ALLOWED_STATS:
                raise ValueError(f"{stat.lower()} not in available Stats")

            stats_func.append(getattr(Stats, stat))

        self.stats = full_stats
        self.element_properties = element_properties
        self.property_names = property_names
        self.n_features = len(property_names)
        self.all_property_names = all_property_names
        self.stats_func = stats_func
        super().__init__(**kwargs)

    def transform_one(self, obj: Union[Structure, str, Composition]) -> pd.DataFrame:
        """
        Transform one object, the object can be string, Compostion or Structure

        Args:
            obj (str/Composition/Structure): object to transform

        Returns: pd.DataFrame with property names as column names

        """
        if isinstance(obj, Structure) or isinstance(obj, Molecule):
            comp = obj.composition
        elif isinstance(obj, str):
            comp = Composition(obj)
        else:
            comp = obj

        element_n_dict = comp.to_data_dict['unit_cell_composition']

        data = []
        weights = []
        for i, j in element_n_dict.items():
            data.append(self.element_properties[i])
            weights.append(j)

        data = list(zip(*data))
        features = []
        for stat in self.stats_func:
            for d in data:
                f = stat(d, weights)
                features.append(f)
        return pd.DataFrame([features], columns=self.all_property_names)

    @classmethod
    def from_file(cls, filename: str, stats: List[str], **kwargs) -> "ElementStats":
        """
        ElementStats from a json file of element property dictionary.
        The keys required are:

            element_properties
            property_names

        Args:
            filename (str): filename
            stats (list): list of stats, check ElementStats.ALLOWED_STATS
                for supported stats. The stats that support additional
                Keyword args, use ':' to separate the args. For example,
                'moment:0:None' will calculate moment stats with order=0,
                and max_order=None.

        Returns: ElementStats class
        """
        with open(filename, "r") as f:
            d = json.load(f)

        property_names = d.get("property_names", None)
        if "element_properties" in d:
            element_properties = d.get("element_properties")
        else:
            element_properties = d
        is_element = _keys_are_elements(element_properties)
        if not is_element:
            raise ValueError("File is not in correct format")

        if 'stats' in d:
            stats = d.get('stats')

        return cls(element_properties=element_properties,
                   property_names=property_names, stats=stats, **kwargs)

    @classmethod
    def from_data(cls, data_name: Union[List[str], str],
                  stats: List[str], num_dim: Optional[int] = None,
                  **kwargs) -> "ElementStats":
        """
        ElementalStats from existing data file. If num_dim is provided,
        PCA dimensional reduction will apply to the elemental properties

        Args:
            data_name (str of list of str): data name. Current supported data are
                available from ElementStats.AVAILABLE_DATA

            stats (list): list of stats, use ElementStats.ALLOWED_STATS to
                check available stats

            num_dim (int): number of dimensions to keep
            **kwargs:

        Returns: ElementStats instance

        """
        if isinstance(data_name, str):
            if data_name not in ElementStats.AVAILABLE_DATA:
                raise ValueError("data name not found in the list %s" % str(ElementStats.AVAILABLE_DATA))
            filename = os.path.join(CWD, DATA_MAPPING[data_name])
            return cls.from_file(filename, stats=stats, **kwargs)

        if len(data_name) == 1:
            return cls.from_data(data_name[0], stats=stats, **kwargs)

        property_names = []

        instances = []
        for data_name_ in data_name:
            instance = cls.from_data(data_name_, stats=stats, **kwargs)
            instances.append(instance)

        elements = [set(i.element_properties.keys()) for i in instances]

        common_keys = elements[0]
        for e in elements[1:]:
            common_keys.intersection_update(e)

        element_properties: Dict = {i: [] for i in common_keys}
        for index, instance in enumerate(instances):
            for k in common_keys:
                element_properties[k].extend(instance.element_properties[k])

            property_names.extend(['%d_%s' % (index, i) for i in instance.property_names])

        if num_dim is not None:
            value_array = []
            p_keys = []
            for i, j in element_properties.items():
                value_array.append(j)
                p_keys.append(i)
            value_np_array = np.array(value_array)
            pca = PCA(n_components=num_dim)

            transformed_values = pca.fit_transform(value_np_array)

            for key, value_list in zip(p_keys, transformed_values):
                element_properties[key] = value_list.tolist()
            property_names = ['pca_%d' % i for i in range(num_dim)]

        return cls(element_properties=element_properties,
                   property_names=property_names,
                   stats=stats,
                   **kwargs)


def _keys_are_elements(dic: Dict) -> bool:
    keys = list(dic.keys())

    for key in keys:
        if not _is_element_or_specie(key):
            return False
    return True


def _is_element_or_specie(s: str) -> bool:
    if s in ['D', 'D+', 'D-', 'T']:
        return True
    try:
        _ = Element(s)
    except ValueError:
        try:
            _ = Specie.from_string(s)
        except ValueError:
            print(s)
            return False
    return True
