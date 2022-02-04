# Pre-trained ML-IAPs from Materials Virtual Lab

This directory contains pre-trained ML-IAPs from the Materials Virtual Lab.

Within `Ni/gap` directory,

```python
from maml.apps.pes._gap import GAPotential
from pymatgen.core import Structure, Element
from maml.apps.pes._lammps import EnergyForceStress, ElasticConstant, DefectFormation

gap = GAPotential.from_config(filename='gap.2020.01.xml')
gap.specie = Element('Ni')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=gap)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is \n {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=gap, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))

defect_calculator = DefectFormation(ff_settings=gap, specie='Ni', lattice='fcc', alat=3.508)
defect_formation_energy = defect_calculator.calculate()
print('The predicted defect formation energy is {} eV'.format(defect_formation_energy))
```

Within `Ni/nnp` directory,

```python
from maml.apps.pes._nnp import NNPotential
from pymatgen.core import Structure
from maml.apps.pes._lammps import EnergyForceStress, ElasticConstant, DefectFormation

nnp = NNPotential.from_config(input_filename='input.nn', scaling_filename='scaling.data', weights_filename='weights.028.data')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=nnp)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is \n {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=nnp, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))

defect_calculator = DefectFormation(ff_settings=nnp, specie='Ni', lattice='fcc', alat=3.508)
defect_formation_energy = defect_calculator.calculate()
print('The predicted defect formation energy is {} eV'.format(defect_formation_energy))
```

Within `Ni/snap` directory,

```python
from maml.apps.pes._snap import SNAPotential
from pymatgen.core import Structure, Element
from maml.apps.pes._lammps import EnergyForceStress, ElasticConstant, DefectFormation

snap = SNAPotential.from_config(coeff_file='SNAPotential.snapcoeff', param_file='SNAPotential.snapparam')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=snap)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is \n {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=snap, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))

defect_calculator = DefectFormation(ff_settings=snap, specie='Ni', lattice='fcc', alat=3.508)
defect_formation_energy = defect_calculator.calculate()
print('The predicted defect formation energy is {} eV'.format(defect_formation_energy))
```

Within `Ni/qsnap` directory,

```python
from maml.apps.pes._snap import SNAPotential
from pymatgen.core import Structure, Element
from maml.apps.pes._lammps import EnergyForceStress, ElasticConstant, DefectFormation

qsnap = SNAPotential.from_config(coeff_file='SNAPotential.snapcoeff', param_file='SNAPotential.snapparam')

Ni_conventional_cell = Structure.from_file('Ni_conventional.cif')
efs_calculator = EnergyForceStress(ff_settings=qsnap)
energy, forces, stresses = efs_calculator.calculate([Ni_conventional_cell])[0]
print('The predicted energy of Ni conventional cell is {} eV'.format(energy))
print('The predicted forces of Ni conventional cell is \n {} eV/Angstrom'.format(forces))

elastic_calculator = ElasticConstant(ff_settings=qsnap, lattice='fcc', alat=3.508)
C11, C12, C44, bulkmodulus = elastic_calculator.calculate()
print('The predicted C11, C12, C44, bulkmodulus are {}, {}, {}, {} GPa'.format(C11, C12, C44, bulkmodulus))

defect_calculator = DefectFormation(ff_settings=qsnap, specie='Ni', lattice='fcc', alat=3.508)
defect_formation_energy = defect_calculator.calculate()
print('The predicted defect formation energy is {} eV'.format(defect_formation_energy))
```
