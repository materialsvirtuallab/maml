from pymatgen.core.structure import Structure


class EnergyModel:
    def predict_energy(self, structure: Structure) -> float:
        """
        Predict the energy of a structure
        Args:
            structure: Pymatgen Structure object.

        Returns: (float) energy value.

        """
        raise NotImplementedError
