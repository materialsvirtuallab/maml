.. image:: https://circleci.com/gh/materialsvirtuallab/maml.svg?style=shield
    :target: https://circleci.com/gh/materialsvirtuallab/maml
.. image:: https://coveralls.io/repos/github/materialsvirtuallab/maml/badge.svg?branch=master
    :target: https://coveralls.io/github/materialsvirtuallab/maml?branch=master

|    

.. image:: ./resources/logo_horizontal.png


maml (MAterials Machine Learning) is a Python package for Materials Machine Learning.

Features
--------

1. Convert materials (crystals and molecules) into features. In addition to common compositional, site and structural features, we provide the following fine-grain local environment features.

 a) Bispectrum coefficients
 b) Behler Parrinello symmetry functions
 c) Smooth Overlap of Atom Position (SOAP)
 d) Graph network features (composition, site and structure)
    
2. Use ML to learn relationship between features and targets. Currently, the `maml` supports `sklearn` and `keras` models. 

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

The SNAP potential comes with this lammps installation. The GAP package for GAP and MLIP package for MTP are needed to run the corresponding potentials. For fitting NNP potential, the `n2p2` package is needed. 

Usage
-----

Many Jupyter notebooks are available on usage. See `notebooks </notebooks>`_.

API documentation
-----------------

See `API docs <https://guide.materialsvirtuallab.org/maml/modules.html>`_.
