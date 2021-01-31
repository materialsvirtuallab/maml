"""
The core classes for XAS prediction app
"""
import os
import warnings

import numpy as np
import pandas as pd
from pymatgen.core.spectrum import Spectrum


class XANES(Spectrum):
    """
    Basic XANES object
    """

    XLABEL = "Energy"
    YLABEL = "Intensity"

    def __init__(self, x, y, absorption_specie, edge, structure=None, e0=None, **kwargs):
        """
        Args:
            x (list/np.array): A sequence of x-ray energies in eV
            y (list/np.array): A sequence of mu(E)
            absorption_specie (str): Specie associated with the XANES.
            edge (str): Spectrum edge, e.g. K, L1, L2, etc
            structure (Structure): Structure associated with the XANES. Pymatgen structure object
            e0 (float): Edge energy associated with the spectrum
            **kwargs: Other attributes of the spectrum object.
        """
        super().__init__(x, y, absorption_specie, edge, structure, e0, **kwargs)
        self.absorption_specie = absorption_specie
        self.edge = edge
        self.structure = structure
        if e0:
            self.e0 = e0
        else:
            warning_msg = "Edge energy is determined with maximum derivative. Using this e0 with caution."
            warnings.warn(warning_msg)
            self.e0 = self.x[np.argmax(np.gradient(self.y) / np.gradient(self.x))]

        for (field, value) in kwargs.items():
            setattr(self, field, value)

    def __str__(self):
        if self.structure:
            return "%s %s Edge for %s: %s" % (
                self.absorption_specie,
                self.edge,
                self.structure.composition.reduced_formula,
                super().__str__(),
            )
        return "%s %s Edge for %s" % (self.absorption_specie, self.edge, super().__str__())

    @staticmethod
    def from_K_XANES_MP_tsv(tsv_file, header=3, absorption_specie=None, **kwargs):
        """
        Args:
            tsv_file: tsv file contains spectrum infor. downloaded from the Materials Project website
            header: number of header row to skip, default to 3
            absorption_specie (str): Specie associated with the XANES. If not given, will be derived using the
                                    edge energy.
            **kwargs: Other attributes of the spectrum object.

        Returns:
            XANES object generated from tsv file

        """

        spectrum_sample = pd.read_csv(tsv_file, sep="\t", header=header)
        spectrum_energy = spectrum_sample["E (eV)"]
        spectrum_mu = spectrum_sample["mu"]

        if absorption_specie is None:
            warning_msg = "Absorption specie is derived using the edge energy determined with maximum derivative."
            warnings.warn(warning_msg)
            edge_energy = spectrum_energy[np.argmax(np.gradient(spectrum_mu) / np.gradient(spectrum_energy))]
            edge_vs_atomNo_filepath = os.path.join(os.path.dirname(__file__), "data", "Edge_vs_atomNo.csv")
            edge_vs_atomNo_df = pd.read_csv(edge_vs_atomNo_filepath)
            kedge_vs_atomNo_df = edge_vs_atomNo_df[["Element", "K Edge"]]
            kedge_vs_atomNo_df["K_edge_difference"] = kedge_vs_atomNo_df["K Edge"].apply(
                lambda x: np.abs(x - edge_energy)
            )
            derived_abs_specie = kedge_vs_atomNo_df.sort_values("K_edge_difference").iloc[0]["Element"]
            return XANES(spectrum_energy, spectrum_mu, absorption_specie=derived_abs_specie, edge="K", **kwargs)

        return XANES(spectrum_energy, spectrum_mu, absorption_specie=absorption_specie, edge="K", **kwargs)
