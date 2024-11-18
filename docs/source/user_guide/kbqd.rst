.. _kbqd:

Kernel-based quadratic distance (KBQD) Goodness-of-Fit tests
=============================================================

The class ``KernelTest`` performs the kernel-based quadratic
distance tests using the Gaussian kernel with bandwidth parameter ``h``.
Depending on the shape of the input ``y`` the function performs the tests
of multivariate normality, non-parametric two-sample tests, or
k-sample tests.

The quadratic distance between two probability distributions :math:`F` and
:math:`G` is defined as

.. math::
   d_{K}(F,G)=\iint K(x,y)d(F-G)(x)d(F-G)(y),

where :math:`G` is a distribution whose goodness of fit we wish to assess and 
:math:`K` denotes the Normal kernel defined as 

.. math::
   K_{{h}}(\mathbf{s}, \mathbf{t}) = (2 \pi)^{-d/2} 
   \left(\det{\mathbf{\Sigma}_h}\right)^{-\frac{1}{2}}  
   \exp\left\{-\frac{1}{2}(\mathbf{s} - \mathbf{t})^\top 
   \mathbf{\Sigma}_h^{-1}(\mathbf{s} - \mathbf{t})\right\},

for every :math:`\mathbf{s}, \mathbf{t} \in \mathbb{R}^d \times 
\mathbb{R}^d`, with covariance matrix :math:`\mathbf{\Sigma}_h=h^2 I` and
tuning parameter :math:`h`.

* **Test for Normality**: 

  Let :math:`x_1, x_2, \ldots, x_n` be a random sample with empirical 
  distribution function :math:`\hat F`. We test the null hypothesis of 
  normality, i.e. :math:`H_0:F=G=\mathcal{N}_d(\mu, \Sigma)`. 

  We consider the U-statistic estimate of the sample KBQD

  .. math::
     U_{n}=\frac{1}{n(n-1)}\sum_{i=2}^{n}\sum_{j=1}^{i-1}
     K_{cen}(\mathbf{x}_{i}, \mathbf{x}_{j}),

  then the first test statistic is 

  .. math::
     T_{n}=\frac{U_{n}}{\sqrt{Var(U_{n})}},

  with :math:`Var(U_n)` computed exactly following Lindsay et al. (2014),
  and the V-statistic estimate 

  .. math::
     V_{n} = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}K_{cen}(\mathbf{x}_{i}, \mathbf{x}_{j}),

  where :math:`K_{cen}` denotes the Normal kernel :math:`K_h` with parametric 
  centering with respect to the considered normal distribution 
  :math:`G = \mathcal{N}_d(\mu, \Sigma)`.

  The asymptotic distribution of the V-statistic is an infinite combination
  of weighted independent chi-squared random variables with one degree of 
  freedom. The cutoff value is obtained using the Satterthwaite 
  approximation :math:`c \cdot \chi_{DOF}^2`, where :math:`c` and :math:`DOF` 
  are computed exactly following the formulas in Lindsay et al. (2014).

  For the :math:`U`-statistic, the cutoff is determined empirically:

  -  Generate data from the considered normal distribution;
  -  Compute the test statistics for ``B`` Monte Carlo (MC) replications;
  -  Compute the 95th quantile of the empirical distribution of the test statistic.

* **k-sample test**:

  Consider :math:`k` random samples of i.i.d. observations 
  :math:`\mathbf{x}^{(i)}_1, \mathbf{x}^{(i)}_{2}, \ldots, \mathbf{x}^{(i)}_{n_i} \sim F_i`, 
  :math:`i = 1, \ldots, k`. We test if the samples are generated from the same *unknown* distribution,
  that is :math:`H_0: F_1 = F_2 = \ldots = F_k` versus 
  :math:`H_1: F_i \not= F_j`, for some :math:`1 \le i \not= j \le k`. 

  We construct a matrix distance :math:`\hat{\mathbf{D}}`, with 
  off-diagonal elements 

  .. math::
     \hat{D}_{ij} = \frac{1}{n_i n_j} \sum_{\ell=1}^{n_i} \sum_{r=1}^{n_j} K_{\bar{F}}(\mathbf{x}^{(i)}_\ell,\mathbf{x}^{(j)}_r), \quad \text{for } i \not= j

  and in the diagonal

  .. math::
     \hat{D}_{ii} = \frac{1}{n_i (n_i -1)} \sum_{\ell=1}^{n_i} \sum_{r\not= \ell}^{n_i} K_{\bar{F}}(\mathbf{x}^{(i)}_\ell,\mathbf{x}^{(i)}_r), \quad \text{for } i = j,

  where :math:`K_{\bar{F}}` denotes the Normal kernel :math:`K_h`
  centered non-parametrically with respect to 

  .. math::
     \bar{F} = \frac{n_1 \hat{F}_1 + \ldots + n_k \hat{F}_k}{n}, \quad \text{with } n=\sum_{i=1}^k n_i.

  We compute the trace statistic

  .. math::
     \mathrm{trace}(\hat{\mathbf{D}}_n) =  \sum_{i=1}^{k}\hat{D}_{ii}

  and :math:`D_n`, derived considering all the possible pairwise comparisons 
  in the *k*-sample null hypothesis, given as

  .. math::
     D_n = (k-1) \mathrm{trace}(\hat{\mathbf{D}}_n) - 2 \sum_{i=1}^{k}\sum_{j > i}^{k}\hat{D}_{ij}.

  We compute the empirical critical value using numerical techniques such as bootstrap, permutation, and subsampling algorithms:

  - Generate k-tuples, of total size :math:`n_B`, from the pooled sample following one of the sampling methods;
  - Compute the k-sample test statistic;
  - Repeat ``B`` times;
  - Select the 95th quantile of the obtained values.

* **Two-sample test**:

  Let :math:`x_1, x_2, \ldots, x_{n_1} \sim F` and 
  :math:`y_1, y_2, \ldots, y_{n_2} \sim G` be
  random samples from the distributions :math:`F` and :math:`G`, respectively.
  We test the null hypothesis that the two samples are generated from 
  the same *unknown* distribution, that is :math:`H_0: F=G` vs 
  :math:`H_1:F\not=G`. The test statistics coincide with the k-sample 
  test statistics when :math:`k=2`.

Kernel Centering
----------------

The arguments :math:`\hat{\mu}` (``mu_hat``) and :math:`\hat{\Sigma}` (``sigma_hat``) indicate the normal model 
considered for the normality test, that is :math:`H_0: F = N(\hat{\mu},\hat{\Sigma})`. 
For the two-sample and k-sample tests, ``mu_hat`` and ``sigma_hat`` can be used for the 
parametric centering of the kernel, in case we wish to 
specify the reference distribution, with ``centering_type = "param"``. 
This is the default method when the test for normality is performed.
The normal kernel centered with respect to :math:`G \sim N_d(\mathbf{\mu}, \mathbf{V})` can be computed as

.. math::
   K_{cen(G)}(\mathbf{s}, \mathbf{t}) = K_{\mathbf{\Sigma_h}}(\mathbf{s}, \mathbf{t}) - K_{\mathbf{\Sigma_h} + \mathbf{V}}(\mathbf{\mu}, \mathbf{t}) 
   - K_{\mathbf{\Sigma_h} + \mathbf{V}}(\mathbf{s}, \mathbf{\mu}) + K_{\mathbf{\Sigma_h} + 2\mathbf{V}}(\mathbf{\mu}, \mathbf{\mu}).

We consider non-parametric centering of the kernel with respect to 
:math:`\bar{F}=(n_1 F_1 + \ldots + n_k F_k)/n` where :math:`n=\sum_{i=1}^k n_i`, 
with ``centering_type = "nonparam"``, for the two- and k-sample 
tests. Let :math:`\mathbf{z}_1,\ldots, \mathbf{z}_n` denote the pooled sample. For any
:math:`s,t \in \{\mathbf{z}_1,\ldots, \mathbf{z}_n\}`, it is given by 

.. math::
   K_{cen(\bar{F})}(\mathbf{s},\mathbf{t}) = K(\mathbf{s},\mathbf{t}) - \frac{1}{n}\sum_{i=1}^{n} K(\mathbf{s},\mathbf{z}_i) - 
   \frac{1}{n}\sum_{i=1}^{n} K(\mathbf{z}_i,\mathbf{t}) + \frac{1}{n(n-1)}\sum_{i=1}^{n} \sum_{j \not=i}^{n} K(\mathbf{z}_i,\mathbf{z}_j).