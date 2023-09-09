# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
"""Setup for maml."""
from __future__ import annotations

from setuptools import find_packages, setup

long_desc = """
maml (MAterials Machine Learning) is a Python package that aims to provide useful high-level interfaces that make ML
for materials science as easy as possible.

The goal of maml is not to duplicate functionality already available in other packages. maml relies on well-established
packages such as scikit-learn and tensorflow for implementations of ML algorithms, as well as other materials science
packages such as `pymatgen <http://pymatgen.org>`_ and `matminer <http://hackingmaterials.lbl.gov/matminer/>`_ for
crystal/molecule manipulation and feature generation.

Official documentation at https://materialsvirtuallab.github.io/maml

Features
--------

1. Convert materials (crystals and molecules) into features. In addition to common compositional, site and structural
   features, we provide the following fine-grain local environment features.

 a) Bispectrum coefficients
 b) Behler Parrinello symmetry functions
 c) Smooth Overlap of Atom Position (SOAP)
 d) Graph network features (composition, site and structure)

2. Use ML to learn relationship between features and targets. Currently, the `maml` supports `sklearn` and `keras`
   models.

3. Applications:

 a) `pes` for modelling the potential energy surface, constructing surrogate models for property prediction.

  i) Neural Network Potential (NNP)
  ii) Gaussian approximation potential (GAP) with SOAP features
  iii) Spectral neighbor analysis potential (SNAP)
  iv) Moment Tensor Potential (MTP)

 b) `rfxas` for random forest models in predicting atomic local environments from X-ray absorption spectroscopy.

Installation
------------

Pip install via PyPI::

    pip install maml

To run the potential energy surface (pes), lammps installation is required you can install from source or from `conda`::

    conda install -c conda-forge/label/cf202003 lammps

The SNAP potential comes with this lammps installation. The GAP package for GAP and MLIP package for MTP are needed to
run the corresponding potentials. For fitting NNP potential, the `n2p2` package is needed.

Install all the libraries from requirement.txt file::

    pip install -r requirements.txt

For all the requirements above::

    pip install -r requirements-ci.txt
    pip install -r requirements-optional.txt
    pip install -r requirements-dl.txt
    pip install -r requirements.txt

Citing
------
::

    @misc{maml,
        author = {Chen, Chi and Zuo, Yunxing and Ye, Weike and Ong, Shyue Ping},
        title = {{Maml - materials machine learning package}},
        year = {2020},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {https://github.com/materialsvirtuallab/maml},
    }

For the ML-IAP package (`maml.pes`), please cite::

    Zuo, Y.; Chen, C.; Li, X.; Deng, Z.; Chen, Y.; Behler, J.; Csányi, G.; Shapeev, A. V.; Thompson, A. P.;
    Wood, M. A.; Ong, S. P. Performance and Cost Assessment of Machine Learning Interatomic Potentials.
    J. Phys. Chem. A 2020, 124 (4), 731-745. https://doi.org/10.1021/acs.jpca.9b08723.
"""

setup(
    name="maml",
    packages=find_packages(),
    version="2023.9.9",
    install_requires=["numpy", "scipy", "monty", "scikit-learn", "pandas", "pymatgen", "tqdm"],
    extras_requires={
        "maml.apps.symbolic._selectors_cvxpy": ["cvxpy"],
        "tensorflow": ["tensorflow>=2"],
        "tensorflow with gpu": ["tensorflow-gpu>=2"],
    },
    author="Materials Virtual Lab",
    author_email="ongsp@eng.ucsd.edu",
    maintainer="Shyue Ping Ong",
    maintainer_email="ongsp@eng.ucsd.edu",
    url="https://materialsvirtuallab.github.io/maml",
    license="BSD",
    description="maml is a machine learning library for materials science.",
    long_description=long_desc,
    keywords=["materials", "science", "deep", "learning"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    include_package_data=True,
    package_data={
        "maml": [
            "describers/data/*.json",
            "describers/data/megnet_models/*.json",
            "describers/data/megnet_mdoels/*.hdf5",
        ]
    },
)
