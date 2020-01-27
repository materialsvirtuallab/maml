"""
Get data from url
"""
from io import StringIO

import requests
import pandas as pd

from maml import DataSource


class URLSource(DataSource):
    """
    Load raw data from a URL, e.g., figshare.
    """

    def __init__(self, fmt: str = "csv", read_kwargs=None):
        self.fmt = fmt
        self.read_kwargs = read_kwargs or {}

    def get(self, url: str) -> pd.DataFrame:
        """
        @param url (str): URL to obtain raw data from.
        @return: pd.DataFrame
        """
        raw = requests.get(url).text
        read = getattr(pd, "read_%s" % self.fmt)
        return read(StringIO(raw), **self.read_kwargs)


class FigshareSource(URLSource):
    """
    Load data from figshare.
    """

    def get(self, file_id: str) -> pd.DataFrame:
        """
        Get the data.

        @param file_id: Figshare file id.
        @return: pandas.DataFrame
        """
        url = "https://ndownloader.figshare.com/files/%s" % file_id
        return super().get(url)
