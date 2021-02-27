.. image:: https://github.com/materialsvirtuallab/maml/blob/master/resources/logo_horizontal.png?raw=true
    :target: https://github.com/materialsvirtuallab/maml
.. image:: https://github.com/materialsvirtuallab/maml/workflows/Testing/badge.svg
    :target: https://github.com/materialsvirtuallab/maml
.. image:: https://github.com/materialsvirtuallab/maml/workflows/Linting/badge.svg
    :target: https://github.com/materialsvirtuallab/maml
.. image:: https://coveralls.io/repos/github/materialsvirtuallab/maml/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/materialsvirtuallab/maml?branch=master
.. image:: https://static.pepy.tech/badge/maml
    :target: https://static.pepy.tech/badge/maml
.. image:: https://img.shields.io/github/commit-activity/y/materialsvirtuallab/maml
    :target: https://github.com/materialsvirtuallab/maml
.. image:: https://img.shields.io/github/repo-size/materialsvirtuallab/maml
    :target: https://github.com/materialsvirtuallab/maml

maml (MAterials Machine Learning) is a Python package that aims to provide useful high-level interfaces that make ML for materials science as easy as possible. 

The goal of maml is not to duplicate functionality already available in other packages. maml relies on well-established packages such as scikit-learn and tensorflow for implementations of ML algorithms, as well as other materials science packages such as `pymatgen <http://pymatgen.org>`_ and `matminer <http://hackingmaterials.lbl.gov/matminer/>`_ for crystal/molecule manipulation and feature generation.

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


Install all the libraries from requirement.txt file::
    pip install -r requirements.txt
(If doesn't works provide the path of requirement.txt file)

For all the requirements above::
    pip install -r requirements-ci.txt
    pip install -r requirements-optional.txt
    pip install -r requirements-tf.txt
    pip install -r requirements.txt
    
    (If doesn't works provide the path of requirements.txt file)

Usage
-----

Many Jupyter notebooks are available on usage. See `notebooks </notebooks>`_. We also have established a tool and a tutorial lecture at nanoHUB `https://nanohub.org/resources/maml <https://nanohub.org/resources/maml>`_.

API documentation
-----------------

See `API docs <https://guide.materialsvirtuallab.org/maml/modules.html>`_.

Machine learning (ML) is the study of computer algorithms that improve automatically through experience.[1][2] It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.[3] Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks.

Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.[5][6] In its application across business problems, machine learning is also referred to as predictive analytics.

Citing
------
:: 

    @misc{maml,
        author = {Chen, Chi and Zuo, Yunxing and Ye, Weike and Ong, Shyue Ping},
        title = {{Maml - materials machine learning package}},
        year = {2020},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/materialsvirtuallab/maml}},
    }

