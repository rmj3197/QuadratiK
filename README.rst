==========
QuadratiK
==========
|GitHub Actions|_ |Codecov|_ |Documentation Status|_ |PyPI Version|_ |PyPI Python Version|_ |PyPI Downloads|_ |GitHub Search|_ |Black|_

.. |GitHub Actions| image:: https://github.com/rmj3197/QuadratiK/actions/workflows/release.yml/badge.svg
   :target: https://github.com/rmj3197/QuadratiK/actions/workflows/release.yml
   :alt: Publish to PyPI

.. |Codecov| image:: https://codecov.io/gh/rmj3197/QuadratiK/graph/badge.svg?token=PPFZDNLJ1N
   :target: https://codecov.io/gh/rmj3197/QuadratiK
   :alt: Codecov

.. |Documentation Status| image:: https://readthedocs.org/projects/quadratik/badge/?version=latest
   :target: https://quadratik.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |PyPI Version| image:: https://img.shields.io/pypi/v/QuadratiK
   :alt: PyPI - Version

.. |PyPI Python Version| image:: https://img.shields.io/pypi/pyversions/QuadratiK
   :alt: PyPI - Python Version

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/QuadratiK
   :alt: PyPI Total Downloads

.. |GitHub Search| image:: https://img.shields.io/github/search/rmj3197/QuadratiK/QuadratiK?logo=github
   :target: https://github.com/rmj3197/QuadratiK
   :alt: GitHub search hit counter

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black Code Formatter

.. _GitHub Actions: https://github.com/rmj3197/QuadratiK/actions/workflows/release.yml
.. _Codecov: https://codecov.io/gh/rmj3197/QuadratiK
.. _Documentation Status: https://quadratik.readthedocs.io/en/latest/?badge=latest
.. _PyPI Version: https://pypi.org/project/QuadratiK/
.. _PyPI Python Version: https://pypi.org/project/QuadratiK/
.. _PyPI Downloads: https://pepy.tech/project/quadratik
.. _GitHub Search: https://github.com/rmj3197/QuadratiK
.. _Black: https://github.com/psf/black

Introduction
==============

The QuadratiK package is implemented in both **R** and **Python**, providing a comprehensive set of goodness-of-fit tests and a clustering technique using kernel-based quadratic distances. This framework aims to bridge the gap between the statistical and machine learning literatures. It includes:

* **Goodness-of-Fit Tests** : The software implements one, two, and k-sample tests for goodness of fit, offering an efficient and mathematically sound way to assess the fit of probability distributions. Expanded capabilities include supporting tests for uniformity on the :math:`d`-dimensional Sphere based on Poisson kernel densities.

* **Clustering Algorithm for Spherical Data**: the package incorporates a unique clustering algorithm specifically tailored for spherical data. This algorithm leverages a mixture of Poisson-kernel-based densities on the sphere, enabling effective clustering of spherical data or data that has been spherically transformed. This facilitates the uncovering of underlying patterns and relationships in the data. Additionally, the package also includes Poisson Kernel-based Densities random number generation.

* **Additional Features**: Alongside these functionalities, the software includes additional graphical functions, aiding users in validating cluster results as well as visualizing and representing clustering results. This enhances the interpretability and usability of the analysis.

* **User Interface**: We also provide a dashboard application built using ``streamlit`` allowing users to access the methods implemented in the package without the need for programming.

Authors
---------
Giovanni Saraceno <gsaracen@buffalo.edu>, Marianthi Markatou <markatou@buffalo.edu>, Raktim Mukhopadhyay <raktimmu@buffalo.edu>, Mojgan Golzy <golzym@health.missouri.edu>

Mantainer: Raktim Mukhopadhyay <raktimmu@buffalo.edu>

Documentation
---------------

The documentation is hosted on Read the Docs at - https://quadratik.readthedocs.io/en/latest/

Installation using ``pip``
----------------------------

``pip install QuadratiK``

Examples
----------

Find basic examples at `QuadratiK Examples <https://quadratik.readthedocs.io/en/latest/user_guide/basic_usage.html>`_

You can also execute the examples on Binder |Binder|. 

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/rmj3197/QuadratiK/HEAD?labpath=doc%2Fsource%2Fuser_guide

Community
------------

Development Version Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For installing the development version, please download the code files from the master branch of the Github repository. 
Please note that installation from Github might be buggy, for latest stable release please download using ``pip``.
For downloading from Github use the following instructions, 

.. code-block:: bash

   git clone https://github.com/rmj3197/QuadratiK.git
   cd QuadratiK
   pip install -e .

Contributing Guide
^^^^^^^^^^^^^^^^^^^^^

Please refer to the `Contributing Guide <https://quadratik.readthedocs.io/en/latest/development/CONTRIBUTING.html>`_.

Code of Conduct
^^^^^^^^^^^^^^^^^

The code of conduct can be found at `Code of Conduct <https://quadratik.readthedocs.io/en/latest/development/CODE_OF_CONDUCT.html>`_. 

License
^^^^^^^^^

This project uses the GPL-3.0 license, with a full version of the license included in the repository `here <https://github.com/rmj3197/QuadratiK/blob/master/LICENSE>`_.

Funding Information
---------------------
The work has been supported by Kaleida Health Foundation, Food and Drug Administration, and Department of Biostatistics, University at Buffalo. 

References
------------
Saraceno G., Markatou M., Mukhopadhyay R., Golzy M. (2024). 
Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python. arXiv preprint arXiv:2402.02290.

Ding Y., Markatou M., Saraceno G. (2023). “Poisson Kernel-Based Tests for
Uniformity on the d-Dimensional Sphere.” Statistica Sinica. DOI: 10.5705/ss.202022.0347.

Golzy M. & Markatou M. (2020) Poisson Kernel-Based Clustering on the Sphere:
Convergence Properties, Identifiability, and a Method of Sampling, Journal of Computational and
Graphical Statistics, 29:4, 758-770, DOI: 10.1080/10618600.2020.1740713.

Markatou M, Saraceno G, Chen Y (2023). “Two- and k-Sample Tests Based on Quadratic Distances.”
Manuscript, (Department of Biostatistics, University at Buffalo).
