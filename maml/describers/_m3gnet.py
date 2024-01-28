from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from maml.base import BaseDescriber, describer_type

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

DEFAULT_MODEL = (
    Path(__file__).parent / "data/m3gnet_models/matbench_mp_e_form/0/m3gnet/"
)


@describer_type("structure")
class M3GNetStructure(BaseDescriber):
    """Use M3GNet pre-trained models as featurizer to get Structural features."""

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
            **kwargs: Pass through to BaseDescriber.
        """
        from m3gnet.models import M3GNet

        if model_path:
            self.describer_model = M3GNet.from_dir(model_path)
            self.model_path = model_path
        else:
            self.describer_model = M3GNet.from_dir(DEFAULT_MODEL)
            self.model_path = str(DEFAULT_MODEL)
        super().__init__(**kwargs)

    def transform_one(self, structure: Structure | Molecule):
        """
        Transform structure/molecule objects into structural features.

        Args:
            structure (Structure/Molecule): target object structure or molecule
        Returns: M3GNet readout layer output as structural features.

        """
        from m3gnet.graph import Index, tf_compute_distance_angle
        from m3gnet.layers import polynomial

        graph = self.describer_model.graph_converter.convert(structure).as_list()
        graph = tf_compute_distance_angle(graph)
        three_basis = self.describer_model.basis_expansion(graph)
        three_cutoff = polynomial(
            graph[Index.BONDS], self.describer_model.threebody_cutoff
        )
        g = self.describer_model.featurizer(graph)
        g = self.describer_model.feature_adjust(g)
        for i in range(self.describer_model.n_blocks):
            g = self.describer_model.three_interactions[i](g, three_basis, three_cutoff)
            g = self.describer_model.graph_layers[i](g)
        layer_before_readout = self.describer_model.layers[-2].layers[0]
        return np.array(layer_before_readout(g))[0]


@describer_type("site")
class M3GNetSite(BaseDescriber):
    """Use M3GNet pre-trained models as featurizer to get atomic features."""

    def __init__(
        self,
        model_path: str | None = None,
        output_layers: list | None = None,
        return_type: list | dict = pd.DataFrame,
        **kwargs,
    ):
        """

        Args:
            model_path (str): m3gnet models path. If no path is provided,
                the models will be M3GNet formation energy model on figshare:
                https://figshare.com/articles/software/m3gnet_property_model_weights/20099465
                Please refer to the M3GNet paper:
                https://doi.org/10.1038/s43588-022-00349-3.
            output_layers: List of names for the layer of GNN as output. Choose from "embedding", "gc_1", "gc_2",...,
                "gc_n", where n is the total number of graph convolutional layers. By default, the node features in
                "gc_1" layer are returned.
            return_type: The data type of the returned the atom features. By default, atom features in different
                output_layers are concatenated to one vector per atom, and a dataframe of vectors are returned.
            **kwargs: Pass through to BaseDescriber. E.g., feature_batch="pandas_concat" is very useful (see test).
        """
        from m3gnet.models import M3GNet

        if model_path:
            self.describer_model = M3GNet.from_dir(model_path)
            self.model_path = model_path
        else:
            self.describer_model = M3GNet.from_dir(DEFAULT_MODEL)
            self.model_path = str(DEFAULT_MODEL)
        allowed_output_layers = ["embedding"] + [
            f"gc_{i + 1}" for i in range(self.describer_model.n_blocks)
        ]
        if output_layers is None:
            output_layers = ["gc_1"]
        elif not isinstance(output_layers, list) or set(output_layers).difference(
            allowed_output_layers
        ):
            raise ValueError(
                f"Invalid output_layers, it must be a sublist of {allowed_output_layers}."
            )
        self.output_layers = output_layers
        self.return_type = return_type
        super().__init__(**kwargs)

    def transform_one(self, structure: Structure | Molecule):
        """
        Transform structure/molecule objects into atom features
        Args:
            structure (Structure/Molecule): target object structure or molecule
        Returns: M3GNet node features as atom features.

        """
        from m3gnet.graph import Index, tf_compute_distance_angle
        from m3gnet.layers import polynomial

        graph = self.describer_model.graph_converter.convert(structure).as_list()
        graph = tf_compute_distance_angle(graph)
        three_basis = self.describer_model.basis_expansion(graph)
        three_cutoff = polynomial(
            graph[Index.BONDS], self.describer_model.threebody_cutoff
        )
        g = self.describer_model.featurizer(graph)
        atom_fea = {"embedding": g[Index.ATOMS]}
        g = self.describer_model.feature_adjust(g)
        for i in range(self.describer_model.n_blocks):
            g = self.describer_model.three_interactions[i](g, three_basis, three_cutoff)
            g = self.describer_model.graph_layers[i](g)
            atom_fea[f"gc_{i + 1}"] = g[Index.ATOMS]
        atom_fea_dict = {k: v for k, v in atom_fea.items() if k in self.output_layers}
        if self.return_type == dict:
            return atom_fea_dict
        return pd.DataFrame(np.concatenate(list(atom_fea_dict.values()), axis=1))
