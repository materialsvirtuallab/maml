# coding: utf-8
# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

import pandas as pd
import numpy as np
from maml.abstract import Describer


class MultiDescriber(Describer):
    """
    This is a generic multiple describer that allows one to combine multiple
    describers.
    """

    def __init__(self, describers):
        """
        :param describers: List of describers. Note that the application of the
            Describers is from left to right. E.g., [Describer1(), Describer2()]
            will run Describer1.describe on the input object, and then run
            Describer2 on the output from Describer1.describe. This provides
            a powerful way to combine multiple describers to generate generic
            descriptors and basis functions.
        """
        self.describers = describers

    def describe(self, obj):
        desc = obj
        for d in self.describers:
            desc = d.describe(desc)
        return desc


class FuncGenerator(Describer):
    """
    General transformer for arrays. In principle, any numerical
    operations can be done as long as each involved function has a
    NumPy.ufunc implementation, e.g., np.sin, np.exp...
    """

    def __init__(self, func_dict, append=True):
        """
        :param func_dict: Dict with labels as keys and stringified
            function as values. The functions arerecovered from strings
            using eval() built-in function. All functions should be
            pointing to a NumPy.ufunc since the calculations will be
            performed on array-like objects. For functions implemented
            elsewhere other than in NumPy, e.g., functions in
            scipy.special, please make sure the module is imported.
        :param append: Whether return the full DataFrame with inputs.
            Default to True.
        """
        self.func_dict = func_dict
        self.append = append

    def describe(self, df):
        """
        Returns description of an object based on all functions.

        :param df: DataFrame with input data.
        :return: DataFrame with transformed data.
        """
        collector = []
        for k, v in self.func_dict.items():
            data = eval(v)(df)
            if isinstance(data, pd.Series):
                data.name = k
            elif isinstance(data, pd.DataFrame):
                columns = [k + " " + c for c in data.columns]
                data.columns = columns
            collector.append(data)
        new_df = pd.concat(collector, axis=1)
        if self.append:
            new_df = df.join(new_df)
        return new_df
