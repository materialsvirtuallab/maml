"""
Utils for describers
"""

import logging
from typing import List, Optional

import numpy as np


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _add_allowed_stats(cls):
    """
    Decorate to add allowed_stats to the Stats class

    Args:
        cls: Stats class

    Returns: Stats class with allowed_stats attributes

    """
    all_keys = list(cls.__dict__.keys())
    allowed = []
    for key in all_keys:
        if isinstance(cls.__dict__[key], staticmethod):
            allowed.append(key)
    setattr(cls, "allowed_stats", allowed)
    return cls


@_add_allowed_stats
class Stats:
    """
    Calculate the stats of a list of values.
    This is particularly helpful when you want to convert
    lists of values of different lengths to uniform length
    for machine learning purposes.

    supported
    """

    @staticmethod
    def max(data: List[float],
            weights: Optional[List[float]] = None) -> float:
        """
        Max of value
        Args:31
            data (list): list of float data
            weights (list): optional weights

        Returns: maximum value

        """
        return np.max(data)

    @staticmethod
    def min(data: List[float],
            weights: Optional[List[float]] = None) -> float:
        """
        min of value
        Args:
            data (list): list of float data
            weights (list): optional weights

        Returns: minimum value

        """
        return np.min(data)

    @staticmethod
    def range(data: List[float],
              weights: Optional[List[float]] = None) -> float:
        """
        Range of values
        Args:
            data (list): list of float data
            weights (list): optional weights

        Returns: range of values, i.e., max - min

        """
        return Stats.max(data) - Stats.min(data)

    @staticmethod
    def moment(data: List[float],
               weights: Optional[List[float]] = None,
               order: Optional[int] = None,
               max_order: Optional[int] = None):
        """
        Moment of probability mass function

        order = 1 means weighted mean
        order = 2 means standard deviation
        order > 2 corresponds to higher order moment to
            the 1./order power

        Args:
            data (list): list of float data
            weights (list or None): weights for each data points
            order (int): moment order
            max_order (int): if set, it will overwrite order

        Returns: float or list of floats

        """
        # check if only a single value should be output
        if max_order is None:
            single = True
        else:
            single = False

        if weights is None:
            weights = [1.0] * len(data)

        if max_order is not None:
            if order is not None:
                logger.info("max_order will overwrite the order parameter")
            orders = list(range(1, max_order + 1))
        else:
            orders = [order or 1]

        stats = [_root_moment(data, weights, i) for i in orders]

        if not single:
            return stats
        return stats[0]

    @staticmethod
    def mean(data: List[float],
             weights: Optional[List[float]] = None) -> float:
        """
        Weighted average

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: average value

        """
        return Stats.moment(data, weights=weights, order=1)

    @staticmethod
    def inverse_mean(data: List[float],
                     weights: Optional[List[float]] = None) -> float:
        """
        inverse mean

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: average value

        """
        return Stats.mean([1. / x for x in data], weights=weights)

    @staticmethod
    def average(data: List[float],
                weights: Optional[List[float]] = None) -> float:
        """
        Weighted average

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: average value

        """
        return Stats.mean(data, weights=weights)

    @staticmethod
    def std(data: List[float],
            weights: Optional[List[float]] = None) -> float:
        """
        Standard deviation

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: Standard deviation

        """

        return Stats.moment(data, weights=weights, order=2)

    @staticmethod
    def skewness(data: List[float],
                 weights: Optional[List[float]] = None) -> float:
        """
        Skewness of the distribution

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: Skewness of the distribution

        """
        third = Stats.moment(data, weights=weights, order=3)
        std = Stats.std(data, weights=weights)
        return third ** 3 / std ** 3

    @staticmethod
    def kurtosis(data: List[float],
                 weights: Optional[List[float]] = None) -> float:
        """
        Kurtosis of the distribution

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: Kurtosis of the distribution

        """
        fourth = Stats.moment(data, weights=weights, order=4)
        std = Stats.std(data, weights=weights)
        return fourth ** 4 / std ** 4

    @staticmethod
    def geometric_mean(data: List[float],
                       weights: Optional[List[float]] = None) -> float:
        """
        Geometric mean of the data

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: geometric mean of the distribution

        """
        return Stats.power_mean(data, weights, p=0)

    @staticmethod
    def power_mean(data: List[float],
                   weights: Optional[List[float]] = None,
                   p: int = 1) -> float:
        """
        power mean https://en.wikipedia.org/wiki/Generalized_mean

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point
            p (int): power

        Returns: power mean of the distribution

        """

        if np.any(np.array(data) <= 0.0):
            raise ValueError("Not possible to calculate geometric means for negative values")

        if weights is None:
            weights = [1. / len(data)] * len(data)

        assert(abs(sum(weights) - 1) < 1e-3)

        if p == 0:
            return np.prod([i ** j for i, j in zip(data, weights)]).item()

        s = np.sum([j * i ** p for i, j in zip(data, weights)])
        return s ** (1. / p)

    @staticmethod
    def shifted_geometric_mean(data: List[float],
                               weights: Optional[List[float]] = None,
                               shift: float = 100) -> float:
        """
        Since we cannot calculate the geometric means on negative or zero values,
        we can first shift all values to positive and then calculate the geometric mean
        afterwards, we shift the computed geometric mean back by a shift value

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point
            shift (float): shift value

        Returns: geometric mean of the distribution

        """
        data_new = [i + shift for i in data]
        return Stats.geometric_mean(data_new, weights=weights) - shift

    @staticmethod
    def harmonic_mean(data: List[float],
                      weights: Optional[List[float]] = None) -> float:
        """
        harmonic mean of the data

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: harmonic mean of the distribution

        """
        if weights is None:
            weights = [1] * len(data)
        return np.sum(weights) / np.sum(np.array(weights) / np.array(data))


def _root_moment(data, weights, order) -> float:
    """
    Auxilliary function to compute moment

    Args:
        data (list): list of float data
        weights (list or None): weights for each data point
        order (int): order of moment

    Returns: moment of order

    """
    weights_sum = np.sum(weights)
    pmf = np.array(weights) / weights_sum
    mean = 0.0

    if order > 1:
        mean = np.mean(data)

    moment = sum([(i - mean) ** order * j for
                  i, j in zip(data, pmf)])

    # when order is odd, moment can be negative
    if moment < 0:
        sign = -1
    else:
        sign = 1

    # avoid error like np.power(-0.001, 1./3.)
    res = (sign * moment) ** (1. / order) * sign
    return res


def _convert_a_or_b(v: str, a=int, b=None):
    try:
        return a(v)
    except ValueError:
        return b


def _moment_symbol_conversion(moment_symbol: str):
    splits = moment_symbol.split(":")
    max_order = _convert_a_or_b(splits[2], int, None)

    if max_order is None:
        return moment_symbol

    if max_order > 0:
        return ['moment:%d:None' % i for i in range(1, max_order + 1)]


def stats_list_conversion(stats_list: List[str]) -> List[str]:
    """
    Convert a list of stats str into a fully expanded list.
    This applies mainly to stats that can return a list of values, e.g.,
    moment with max_order > 1

    Args:
        stats_list (list): list of stats str

    Returns: list of expanded stats str

    """
    re_list = []
    for st in stats_list:
        if ":" not in st:
            re_list.append(st)
        elif 'moment' in st:
            moment = _moment_symbol_conversion(st)
            if isinstance(moment, list):
                re_list.extend(moment)
            else:
                re_list.append(moment)
    return re_list


STATS_KWARGS = {'moment': [{'order': int}, {'max_order': int}]}
