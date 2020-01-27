"""
Methods for data retrieval from various sources. For ease of
"""

import pandas as pd
from pymatgen.ext.matproj import MPRester

def get_mp(criteria, properties):
    mpr = MPRester