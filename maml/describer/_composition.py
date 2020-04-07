"""
Compositional describers
"""
from functools import partial
import os
from typing import Dict, List, Union, Optional

from matminer.featurizers.composition import ElementProperty as MatminerElementProperty  # noqa
from ._matminer_wrapper import wrap_matminer_describer
from pymatgen.core import Composition, Structure, Element
import pandas as pd
import json

from maml.base import BaseDescriber, OutDataFrameConcat
from maml.utils import Stats, STATS_KWARGS, stats_list_conversion


CWD = os.path.abspath(os.path.dirname(__file__))

DATA_MAPPING = {'megnet_1': 'data/elemental_embedding_1MEGNet_layer.json',
                'megnet_3': 'data/elemental_embedding_3MEGNet_layer.json'}


ElementProperty = wrap_matminer_describer("ElementProperty", MatminerElementProperty)


class ElementStats(OutDataFrameConcat, BaseDescriber):
    """
    Element statistics. The allowed stats are accessed via ALLOWED_STATS class
    attributes. If the stats have multiple parameters, the positional arguments
    are separated by ::, e.g., moment::1::None
    """

    ALLOWED_STATS = Stats.allowed_stats  # type: ignore

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
        if isinstance(obj, Structure):
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
                if isinstance(f, list):
                    features.extend(f)
                else:
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

        if stats is None:
            raise ValueError("stats not available")
        return cls(element_properties=element_properties,
                   property_names=property_names, stats=stats, **kwargs)

    @classmethod
    def from_data(cls, data_name: Union[List[str], str],
                  stats: List[str], **kwargs) -> "ElementStats":
        """
        ElementalStats from existing data file
        Args:
            data_name (str of list of str): data name. Current supported data are

                megnet_1: megnet elemental embedding from 1 megnet layer
                megnet_3: megnet elemental embedding from 3 megnet layer

            stats (list): list of stats, use ElementStats.ALLOWED_STATS to
                check available stats
            **kwargs:

        Returns: ElementStats instance

        """
        if isinstance(data_name, str):
            filename = os.path.join(CWD, DATA_MAPPING[data_name])
            return cls.from_file(filename, stats=stats, **kwargs)
        else:
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

            return cls(element_properties=element_properties,
                       property_names=property_names,
                       stats=stats,
                       **kwargs)


def _keys_are_elements(dic: Dict) -> bool:
    keys = list(dic.keys())
    try:
        for key in keys:
            _ = Element(key)
        return True
    except ValueError:
        return False
