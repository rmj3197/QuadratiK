.. _pktest:

Poisson kernel-based quadratic distance test of Uniformity on the sphere
=========================================================================

Let :math:`x_1, x_2, \ldots, x_n` be a random sample with empirical distribution
function :math:`\hat F`. We test the null hypothesis of uniformity on the 
:math:`(d-1)`-dimensional sphere, i.e., :math:`H_0: F = G`, where :math:`G` is the 
uniform distribution on the :math:`(d-1)`-dimensional sphere 
:math:`\mathcal{S}^{d-1}`. 

We compute the U-statistic estimate of the sample KBQD (Kernel-Based 
Quadratic Distance):

.. math::
   U_{n} = \frac{1}{n(n-1)} \sum_{i=2}^{n} \sum_{j=1}^{i-1} K_{cen}
   (\mathbf{x}_{i}, \mathbf{x}_{j}),

then the first test statistic is given as:

.. math::
   T_{n} = \frac{U_{n}}{\sqrt{Var(U_{n})}},

with:

.. math::
   Var(U_{n}) = \frac{2}{n(n-1)}
   \left[\frac{1+\rho^{2}}{(1-\rho^{2})^{d-1}} - 1\right],

and the V-statistic estimate of the KBQD:

.. math::
   V_{n} = \frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{n} K_{cen}
   (\mathbf{x}_{i}, \mathbf{x}_{j}),

where :math:`K_{cen}` denotes the Poisson kernel :math:`K_\rho` centered with
respect to the uniform distribution on the :math:`(d-1)`-dimensional sphere, 
that is:

.. math::
   K_{cen}(\mathbf{u}, \mathbf{v}) = K_\rho(\mathbf{u}, \mathbf{v}) - 1

and:

.. math::
   K_\rho(\mathbf{u}, \mathbf{v}) = \frac{1-\rho^{2}}{\left(1+\rho^{2}-
   2\rho (\mathbf{u} \cdot \mathbf{v})\right)^{d/2}},

for every :math:`\mathbf{u}, \mathbf{v} \in \mathcal{S}^{d-1} 
\times \mathcal{S}^{d-1}`.

The asymptotic distribution of the V-statistic is an infinite combination
of weighted independent chi-squared random variables with one degree of 
freedom. The cutoff value is obtained using the Satterthwaite approximation 
:math:`c \cdot \chi_{DOF}^2`, where:

.. math::
   c = \frac{(1+\rho^{2}) - (1-\rho^{2})^{d-1}}{(1+\rho)^{d} - (1-\rho^{2})^{d-1}}

and:

.. math::
   DOF(K_{cen}) = \left(\frac{1+\rho}{1-\rho}\right)^{d-1}\left\{ 
   \frac{\left(1+\rho - (1-\rho)^{d-1}\right)^{2}}
   {1+\rho^{2} - (1-\rho^{2})^{d-1}}\right\}.

For the :math:`U`-statistic, the cutoff is determined empirically:

- Generate data from a Uniform distribution on the :math:`d`-dimensional sphere;
- Compute the test statistics for ``num_iter`` Monte Carlo (MC) replications;
- Compute the 95th quantile of the empirical distribution of the test
  statistic.
