.. image:: https://github.com/rmj3197/QuadratiK/actions/workflows/release.yml/badge.svg
   :target: https://github.com/rmj3197/QuadratiK/actions/workflows/release.yml
   :alt: Publish to PyPI


QuadratiK
=========

Introduction
-------------

The QuadratiK package is implemented in both **R** and **Python**, providing a comprehensive set of goodness-of-fit tests and a clustering technique using kernel-based quadratic distances. This framework aims to bridge the gap between the statistical and machine learning literatures. It includes:

* **Goodness-of-Fit Tests** : The software implements one, two, and k-sample tests for goodness of fit, offering an efficient and mathematically sound way to assess the fit of probability distributions. Expanded capabilities include supporting tests for uniformity on the :math:`d`-dimensional Sphere based on Poisson kernel densities.

* **Clustering Algorithm for Spherical Data**: the package incorporates a unique clustering algorithm specifically tailored for spherical data. This algorithm leverages a mixture of Poisson-kernel-based densities on the sphere, enabling effective clustering of spherical data or data that has been spherically transformed. This facilitates the uncovering of underlying patterns and relationships in the data.

* **Additional Features**: Alongside these functionalities, the software includes additional graphical functions, aiding users in validating cluster results as well as visualizing and representing clustering results. This enhances the interpretability and usability of the analysis.

The documentation can be found at: `Documentation`_.

.. _Documentation: doc/build/latex/quadratik.pdf

Funding Information
++++++++++++++++++++
The work has been supported by Kaleida Health Foundation, Food and Drug Administration, and Department of Biostatistics, University at Buffalo. 

Authors
++++++++
Giovanni Saraceno <gsaracen@buffalo.edu>, Marianthi Markatou <markatou@buffalo.edu>, Raktim Mukhopadhyay <raktimmu@buffalo.edu>, Mojgan Golzy <golzym@health.missouri.edu>

Mantainer: Raktim Mukhopadhyay <raktimmu@buffalo.edu>

References
+++++++++++
Saraceno G., Markatou M., Mukhopadhyay R., Golzy M. (2023). Goodness of-
fit and clustering of spherical data: The QuadratiK package in R and Python. Technical Report,Department of Biostatistics, University at Buffalo.

Ding Y., Markatou M., Saraceno G. (2023). “Poisson Kernel-Based Tests for
Uniformity on the d-Dimensional Sphere.” Statistica Sinica. doi: doi:10.5705/ss.202022.0347.

Golzy M. & Markatou M. (2020) Poisson Kernel-Based Clustering on the Sphere:
Convergence Properties, Identifiability, and a Method of Sampling, Journal of Computational and
Graphical Statistics, 29:4, 758-770, DOI: 10.1080/10618600.2020.1740713.

Markatou M, Saraceno G, Chen Y (2023). “Two- and k-Sample Tests Based on Quadratic Distances.”
Manuscript, (Department of Biostatistics, University at Buffalo).
