"""CGCNN Wrapper."""
import argparse
import os
import warnings
from typing import Dict, List, Tuple
import numpy as np

from pymatgen import Structure
from bowsr.model.base import EnergyModel

try:
    import torch
    from cgcnn.model import CrystalGraphConvNet
    from cgcnn.data import GaussianDistance, AtomCustomJSONInitializer
except ImportError:
    print("Please refer to https://github.com/txie-93/cgcnn.git to download package of cgcnn")

pjoin = os.path.join
module_dir = os.path.dirname(__file__)
model_dir = pjoin(module_dir, "model_files", "cgcnn", "formation-energy-per-atom.pth.tar")


class CGCNN(EnergyModel):
    """Wrapper to generate cgcnn energy prediction model."""

    def __init__(self,
                 model_path: str = model_dir,
                 orig_atom_fea_len: int = 92,
                 nbr_fea_len: int = 41):

        """
        Init CGCNN.
        Args:
            model_path(str): path of model
            orig_atom_fea_len(int): Number of atom features in the input.
                                    i.e. Original atom feature length
                                    (default 92)
            nbr_fea_len(int): Number of bond features.
                            i.e. Number of neighbors (default 41)

        """
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**checkpoint['args'])
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            n_h=model_args.n_h,
            n_conv=model_args.n_conv,
            h_fea_len=model_args.h_fea_len)
        self.normalizer = CGCNNNormalizer(torch.zeros(3))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.normalizer.load_state_dict(checkpoint['normalizer'])

    def predict_energy(self, structure: Structure) -> np.ndarray:
        """
        CGCNN predict formatio nenergy from pymatgen structure.

        Args:
            structure(Structure): structure to be predicted

        Returns: formation energy (eV/atom) of provided structure

        """
        self.model.eval()
        input_generator = CGCNNInput()
        inp = input_generator.generate_input(structure)
        inp = inp + ([torch.LongTensor(np.arange(structure.num_sites))],)
        output = self.model(*inp)
        return self.normalizer.denorm(output).data.cpu().numpy()[0][0]


class CGCNNInput:
    """Wrapper to generate input for cgcnn from pymatgen structure."""

    atom_init_filename = pjoin(module_dir,
                               "model_files", "cgcnn", "atom_init.json")

    def __init__(self,
                 max_num_nbr: int = 12,
                 radius: float = 8,
                 dmin: float = 0,
                 step: float = 0.2):

        """
        Init CGCNNInput.

        Args:
            max_num_nbr(int): The maximum number of neighbors
                            while constructing the crystal graph
                            (default 12)
            radius(float): The cutoff radius for searching neighbors
                            (default 8)
            dmin(float): The minimum distance for constructing
                        GaussianDistance (default 0)
            step(float): The step size for constructing GaussianDistance
                        (default 0.2)
        """
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.ari = AtomCustomJSONInitializer(self.atom_init_filename)

    def _get_nbr_fea(self, all_nbrs: list, cif_id: int) -> Tuple[np.ndarray, ...]:
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        return tuple((nbr_fea_idx, nbr_fea))

    def generate_input(self,
                       structure: Structure,
                       cif_id: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:

        """
        Generate cgcnn inputs for given structure.

        Args:
            structure(Structure): structure to get input for
            cif_id(int): Optional, the id of the structure

        Returns: Tuple of input (atom_fea, nbr_fea, nbr_fea_idx)
        """
        atom_fea = [sum([(self.ari.get_atom_fea(el.Z) * oc)
                         for el, oc in site.species.items()])
                    for site in structure]
        atom_fea = np.vstack(atom_fea)
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        # sort the nbrs by distance
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        if cif_id:
            nbr_fea_idx, nbr_fea = self._get_nbr_fea(all_nbrs, cif_id)
        else:
            nbr_fea_idx, nbr_fea = self._get_nbr_fea(all_nbrs, 0)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        return tuple((atom_fea, nbr_fea, nbr_fea_idx))

    def generate_inputs(self,
                        structures: List[Structure],
                        cif_ids: List[int] = None) \
            -> List[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]]:
        """
        Generate cgcnn inputs for given list of structures
        Args:
            structures (list): List of structures to get inputs for.
            cif_ids (list): Optional, the list of ids of the structures.

        """
        if not cif_ids:
            cif_ids = range(len(structures))
        return [self.generate_input(s, id) for s, id in zip(structures, cif_ids)]


class CGCNNNormalizer(object):

    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor: torch.Tensor):

        """
        Tensor is taken as a sample to calculate the mean and std.

        Args:
            tensor(torch.Tensor): data
        """
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:

        """
        Normalize tensor.

        Args:
            tensor(torch.Tensor): data

        Returns: normalized tensor

        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:

        """
        Denormalize tensor.

        Args:
            normed_tensor(torch.Tensor): normalized tensor data

        Returns: denormalized tensor

        """
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> dict:

        """
        Get dict of mean and std.

        Returns: dict of mean and std of the normalizer。

        """
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict: Dict) -> None:

        """
        Load the normalizer with mean and std.

        Args:
            state_dict(Dict): dict of mean and std

        Returns: None

        """
        self.mean = state_dict['mean']
        self.std = state_dict['std']
