"""
Materials Project DataSource
"""

import pandas as pd
from pymatgen.ext.matproj import MPRester

from maml import DataSource


class MaterialsProject(DataSource):
    """
    Query the Materials Project for Data.
    """

    def __init__(self, api_key=None):
        """
        :param api_key: Materials Project API key.
        """
        self.mpr = MPRester(api_key=api_key)

    def get(self, criteria, properties):
        """
        :param criteria: Criteria for query
        :param properties: Properties to be queried.
        """
        data = self.mpr.query(criteria=criteria, properties=properties)
        return pd.DataFrame(data)
