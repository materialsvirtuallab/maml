"""
MAML data base classes
"""
import abc

import pandas as pd


class BaseDataSource:
    """
    Abstract base class for a data source.
    """

    @abc.abstractmethod
    def get(self, *args, **kwargs) -> pd.DataFrame:
        """
        Get data from sources
        """
