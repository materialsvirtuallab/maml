"""
magpie model wrapper
"""
try:
    import pickle
    import numpy as np
    from matminer.featurizers.composition import ElementProperty
except Exception as error:
    raise ImportError("magpie module is not installed!")

import os
from pymatgen.core.structure import Structure
from maml.apps.bowsr.model.base import EnergyModel

module_dir = os.path.dirname(__file__)
module_filename = os.path.join(module_dir, "model_files", "magpie", "magpie.pkl")


class Magpie(EnergyModel):
    """
    Magpie model.
    """

    def __init__(self):
        """
        Magpie constructor
        """
        with open(module_filename, "rb") as f:
            model = pickle.load(f)
        self.model = model
        self.featurizer = ElementProperty.from_preset("magpie")

    def predict_energy(self, structure: Structure) -> float:
        """
        Predict energy from structure.

        Args:
            structure: (pymatgen Structure)
        """
        return self.model.predict(np.reshape(self.featurizer.featurize(structure.composition), [1, -1]))[0]
