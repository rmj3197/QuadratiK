.. _pkbd:

An Introduction to Poisson Kernel-Based Distributions
=======================================================

The Poisson kernel-based densities are based on the normalized Poisson kernel
and are defined on the :math:`d`-dimensional unit sphere. Given a vector 
:math:`\mathbf{\mu} \in \mathcal{S}^{d-1}`, where :math:`\mathcal{S}^{d-1}= 
\{x \in \mathbb{R}^d : ||x|| = 1\}`, and a parameter :math:`\rho` such that 
:math:`0 < \rho < 1`, the probability density function of a :math:`d`-variate 
Poisson kernel-based density is defined by:

.. math::

   f(\mathbf{x}|\rho, \mathbf{\mu}) = \frac{1-\rho^2}{\omega_d 
   ||\mathbf{x} - \rho \mathbf{\mu}||^d},

where :math:`\mu` is a vector orienting the center of the distribution, 
:math:`\rho` is a parameter to control the concentration of the distribution 
around the vector :math:`\mu`, and it is related to the variance of the 
distribution. Recall that, for :math:`x = (x_1, \ldots, x_d) \in \mathbb{R}^d`,
:math:`||x|| = \sqrt{x_1^2 + \ldots + x_d^2}`. Furthermore, :math:`\omega_d =
2\pi^{d/2} [\Gamma(d/2)]^{-1}` is the surface area of the unit sphere in
:math:`\mathbb{R}^d` (see Golzy and Markatou, 2020). When :math:`\rho \to 0`, 
the Poisson kernel-based density tends to the uniform density on the sphere.

The connection of the PKBDs to other distributions is discussed in detail in 
Golzy and Markatou (2020). Here we note that when :math:`d=2`, PKBDs reduce to 
the wrapped Cauchy distribution. Additionally, with precise choice of the 
parameters :math:`\rho` and :math:`\mu`, the two-dimensional PKBD becomes a 
two-dimensional projected normal distribution. However, the connection with 
:math:`d`-dimensional projected normal distributions does not extend beyond 
:math:`d=2`.

Golzy and Markatou (2020) proposed an acceptance-rejection method for 
simulating data from a PKBD using von Mises-Fisher envelopes (:code:`rejvmf`). 
Furthermore, Sablica, Hornik, and Leydold (2023) proposed new methods 
for simulating from the PKBD, using angular central Gaussian envelopes 
(:code:`rejacg`).

Please see :py:mod:`QuadratiK.spherical_clustering.PKBD` for details on using the
random sample generation and density estimation functions of the PKB distribution.
Usage examples are included in `User Guide <user_guide>`_.

References
************

Golzy, M., & Markatou, M. (2020). Poisson Kernel-Based Clustering on the Sphere: Convergence Properties, Identifiability, and a Method of Sampling. Journal of Computational and Graphical Statistics, 29(4), 758–770. https://doi.org/10.1080/10618600.2020.1740713

Sablica, L., Hornik, K., & Leydold, J. (2023). Efficient sampling from the PKBD distribution. Electronic Journal of Statistics, 17(2), 2180-2209.