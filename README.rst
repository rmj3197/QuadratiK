==========
QuadratiK
==========

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - **Usage**
     - **Release**
     - **Development**
   * - |License|_ |PyPI Python Version|_ |PyPI Downloads|_
     - |PyPI Version|_ |GitHub Actions|_ |Documentation Status|_
     - |Codecov|_ |Ruff|_ |Black|_ |Codacy|_ |Codefactor|_ |Repo Status|_

.. |License| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://github.com/rmj3197/QuadratiK/blob/main/LICENSE
   :alt: License: GPL v3

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

.. |Black| image:: https://github.com/rmj3197/QuadratiK/actions/workflows/black_check.yml/badge.svg
   :target: https://github.com/rmj3197/QuadratiK/actions/workflows/black_check.yml
   :alt: Black

.. |Ruff| image:: https://github.com/rmj3197/QuadratiK/actions/workflows/ruff_linting.yml/badge.svg
   :target: https://github.com/rmj3197/QuadratiK/actions/workflows/ruff_linting.yml
   :alt: Ruff Linting

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/321a7de540c5458da777ff883f81812f
   :target: https://app.codacy.com/gh/rmj3197/QuadratiK/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
   :alt: Codacy Badge

.. |Codefactor| image:: https://www.codefactor.io/repository/github/rmj3197/quadratik/badge
   :target: https://www.codefactor.io/repository/github/rmj3197/quadratik
   :alt: CodeFactor

.. |Repo Status| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. _GitHub Actions: https://github.com/rmj3197/QuadratiK/actions/workflows/release.yml
.. _Codecov: https://codecov.io/gh/rmj3197/QuadratiK
.. _Documentation Status: https://quadratik.readthedocs.io/en/latest/?badge=latest
.. _PyPI Version: https://pypi.org/project/QuadratiK/
.. _PyPI Python Version: https://pypi.org/project/QuadratiK/
.. _PyPI Downloads: https://pepy.tech/project/quadratik
.. _Black: https://github.com/psf/black
.. _Repo Status: https://www.repostatus.org/#active
.. _Ruff: https://github.com/rmj3197/QuadratiK/actions/workflows/ruff_linting.yml
.. _Codacy: https://app.codacy.com/gh/rmj3197/QuadratiK/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
.. _Codefactor: https://www.codefactor.io/repository/github/rmj3197/quadratik

Introduction
==============

The QuadratiK package is implemented in both **R** and **Python**, providing a comprehensive set of goodness-of-fit tests and a clustering technique using kernel-based quadratic distances. This framework aims to bridge the gap between the statistical and machine learning literatures. It includes:

* **Goodness-of-Fit Tests** : The software implements one, two, and k-sample tests for goodness of fit, offering an efficient and mathematically sound way to assess the fit of probability distributions. Expanded capabilities include supporting tests for uniformity on the :math:`d`-dimensional Sphere based on Poisson kernel densities.

* **Clustering Algorithm for Spherical Data**: the package incorporates a unique clustering algorithm specifically tailored for spherical data. This algorithm leverages a mixture of Poisson-kernel-based densities on the sphere, enabling effective clustering of spherical data or data that has been spherically transformed. This facilitates the uncovering of underlying patterns and relationships in the data. Additionally, the package also includes Poisson Kernel-based Densities random number generation.

* **Additional Features**: Alongside these functionalities, the software includes additional graphical functions, aiding users in validating cluster results as well as visualizing and representing clustering results. This enhances the interpretability and usability of the analysis.

* **User Interface**: We also provide a dashboard application built using ``streamlit`` allowing users to access the methods implemented in the package without the need for programming.

The **R** implementation is also available on `CRAN <https://cran.r-project.org/web/packages/QuadratiK/index.html>`_.

Authors
---------
Giovanni Saraceno <gsaracen@buffalo.edu>, Marianthi Markatou <markatou@buffalo.edu>, Raktim Mukhopadhyay <raktimmu@buffalo.edu>, Mojgan Golzy <golzym@health.missouri.edu>

Mantainer: Raktim Mukhopadhyay <raktimmu@buffalo.edu>

Documentation
===============

The documentation is hosted on Read the Docs at - https://quadratik.readthedocs.io/en/latest/

Installation using ``pip``
============================

The package can be installed from PyPI using ``pip install QuadratiK``

Usage Examples
===============

- `QuadratiK Examples <https://quadratik.readthedocs.io/en/latest/user_guide/basic_usage.html>`_:
  A collection of basic examples that demonstrate how to use the core functionalities of the QuadratiK package. Ideal for new users to get started quickly.

- `Random sampling from the Poisson kernel-based density <https://quadratik.readthedocs.io/en/latest/user_guide/gen_plot_rpkb.html>`_:
  Learn how to generate random samples from the Poisson kernel-based density and visualize the results.

- `Usage Instructions for Dashboard Application <https://quadratik.readthedocs.io/en/latest/user_guide/dashboard_application_usage.html>`_:
  Step-by-step instructions on how to set up and use the QuadratiK dashboard application. This guide helps you interactively explore and analyze data using the dashboard's features.

You can also execute the examples on Binder |Binder|. 

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/rmj3197/QuadratiK/HEAD?labpath=doc%2Fsource%2Fuser_guide

Community
===========

Development Version Installation
----------------------------------

For installing the development version, please download the code files from the master branch of the Github repository. 
Please note that installation from Github might be buggy, for latest stable release please download using ``pip``.
For downloading from Github use the following instructions, 

.. code-block:: bash

   git clone https://github.com/rmj3197/QuadratiK.git
   cd QuadratiK
   pip install -e .

Contributing Guide
---------------------

Please refer to the `Contributing Guide <https://quadratik.readthedocs.io/en/latest/development/CONTRIBUTING.html>`_.

Code of Conduct
----------------

The code of conduct can be found at `Code of Conduct <https://quadratik.readthedocs.io/en/latest/development/CODE_OF_CONDUCT.html>`_. 

License
--------

This project uses the GPL-3.0 license, with a full version of the license included in the repository `here <https://github.com/rmj3197/QuadratiK/blob/master/LICENSE>`_.


Citation
==========

If you use this package, please consider citing it using the following entry:

.. code-block:: tex

    @misc{saraceno2024goodnessoffitclusteringsphericaldata,
          title={Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python}, 
          author={Giovanni Saraceno and Marianthi Markatou and Raktim Mukhopadhyay and Mojgan Golzy},
          year={2024},
          eprint={2402.02290},
          archivePrefix={arXiv},
          primaryClass={stat.CO},
          url={https://arxiv.org/abs/2402.02290}, 
    }

Related Packages
=================

Below is a list of packages in `R` and `Python` that provide functionalities related to Goodness-of-Fit testing. 
Please note that this list is not exhaustive.

R Packages
------------

- ``stats``: Contains the Kolmogorov-Smirnov test, performed using the `ks.test` function.
- ``goftest``: Includes the Cramér-von Mises test.
- ``goft``: Provides the Anderson-Darling test.
- ``vsgoftest``: Performs GoF tests for various distributions (uniform, normal, lognormal, exponential, gamma, Weibull, Pareto, Fisher, Laplace, and Beta) based on Shannon entropy and the Kullback-Leibler divergence.
- ``GoFKernel``: Contains an implementation of Fan's test.
- ``GSAR``: Implements graph-based ranking strategies for univariate and high-dimensional multivariate two-sample GoF tests. Includes the univariate run-based test, two-sample Kolmogorov-Smirnov test, and a modified Kolmogorov-Smirnov test for scale alternatives.
- ``crossmatch``: Provides a two-sample test based on interpoint distances.
- ``energy``: Offers a collection of test statistics for multivariate inference based on energy statistics.
- ``kernlab``: Includes an implementation of the Maximum Mean Discrepancy (MMD) test statistic using kernel mean embedding properties.
- ``kSamples``: Contains several nonparametric Rank Score $k$-sample tests, including the Kruskal-Wallis test, van der Waerden scores, normal scores, and the Anderson-Darling test.
- ``coin``: Provides permutation tests tailored against location and scale alternatives, and for survival distributions.
- ``circular``: Offers tests for data represented as points on the surface of a unit hypersphere, including Rayleigh's test, Rao’s Spacing test, Kuiper's test, and Watson's test of uniformity.
- ``CircNNTSR``: Provides a test for uniformity based on nonnegative trigonometric sums.
- ``sphunif``: Contains a collection of Sobolev tests and other nonparametric tests for uniformity on the sphere.

Python Packages
---------------

- ``scipy``: Includes a number of goodness-of-fit (GoF) tests, such as the Kolmogorov-Smirnov test, Cramér-von Mises test, and Anderson-Darling test. For more details, please see the `Scipy Statistical Functions documentation <https://docs.scipy.org/doc/scipy/reference/stats.html>`_.
- ``hyppo``: This package offers implementations of various Goodness-of-Fit (GoF) testing methods, such as the Maximum Mean Discrepancy (MMD) and Energy statistics for $k$-sample testing. For more information, visit: `Hyppo Documentation <https://hyppo.neurodata.io/>`_.


Funding Information
=====================
The work has been supported by Kaleida Health Foundation, Food and Drug Administration, and Department of Biostatistics, University at Buffalo. 

References
============
Saraceno G., Markatou M., Mukhopadhyay R., Golzy M. (2024). 
Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python. arXiv preprint arXiv:2402.02290.

Ding Y., Markatou M., Saraceno G. (2023). “Poisson Kernel-Based Tests for
Uniformity on the d-Dimensional Sphere.” Statistica Sinica. DOI: 10.5705/ss.202022.0347.

Golzy M. & Markatou M. (2020) Poisson Kernel-Based Clustering on the Sphere:
Convergence Properties, Identifiability, and a Method of Sampling, Journal of Computational and
Graphical Statistics, 29:4, 758-770, DOI: 10.1080/10618600.2020.1740713.

Markatou M, Saraceno G, Chen Y (2023). “Two- and k-Sample Tests Based on Quadratic Distances.”
Manuscript, (Department of Biostatistics, University at Buffalo).
