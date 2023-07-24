from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from maml.base import BaseDescriber

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

DEFAULT_MODEL = Path(__file__).parent / "data/m3gnet_models/matbench_mp_e_form/0/m3gnet/"


class M3GNetStructure(BaseDescriber):
    def __init__(
        self,
        model_path: str | None = None,
        **kwargs,
    ):
        """
        Args:
            model_path (str): m3gnet models path. If no path is provided,
            the models will be M3GNet formation energy model on figshare:
            https://figshare.com/articles/software/m3gnet_property_model_weights/20099465
            Please refer to the M3GNet paper:
            https://doi.org/10.1038/s43588-022-00349-3.
        """
        from m3gnet.models import M3GNet

        if model_path:
            self.describer_model = M3GNet.from_dir(model_path)
        else:
            self.describer_model = M3GNet.from_dir(DEFAULT_MODEL)
        self.model_path = model_path
        super().__init__(**kwargs)

    def transform_one(self, structure: Structure | Molecule):
        """
        Transform structure/molecule objects into features
        Args:
            structure (Structure/Molecule): target object structure or molecule
        Returns: np.array features.

        """
        from m3gnet.graph import Index, tf_compute_distance_angle
        from m3gnet.layers import polynomial

        graph = self.describer_model.graph_converter.convert(structure).as_list()
        graph = tf_compute_distance_angle(graph)
        three_basis = self.describer_model.basis_expansion(graph)
        three_cutoff = polynomial(graph[Index.BONDS], self.describer_model.threebody_cutoff)
        g = self.describer_model.featurizer(graph)
        g = self.describer_model.feature_adjust(g)
        for i in range(self.describer_model.n_blocks):
            g = self.describer_model.three_interactions[i](g, three_basis, three_cutoff)
            g = self.describer_model.graph_layers[i](g)
        layer_before_readout = self.describer_model.layers[-2].layers[0]
        return np.array(layer_before_readout(g))[0].tolist()
