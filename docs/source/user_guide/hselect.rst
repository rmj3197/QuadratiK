.. _hselect:

Select the value of the kernel tuning parameter (h)
===================================================

The function performs the selection of the optimal value for the tuning 
parameter :math:`h` of the normal kernel function, for normality test, the 
two-sample, and k-sample KBQD tests. It performs a small simulation study,
generating samples according to the family of `alternative` specified, 
for the chosen values of `h_values` and `delta`.

We consider target alternatives :math:`F_\delta(\hat{\mathbf{\mu}},
\hat{\mathbf{\Sigma}}, \hat{\mathbf{\lambda}})`, where 
:math:`\hat{\mathbf{\mu}}, \hat{\mathbf{\Sigma}}` and 
:math:`\hat{\mathbf{\lambda}}` indicate the location,
covariance, and skewness parameter estimates from the pooled sample. 

- Compute the estimates of the mean :math:`\hat{\mu}`, covariance matrix
  :math:`\hat{\Sigma}`, and skewness :math:`\hat{\lambda}` from the pooled sample.
- Choose the family of alternatives :math:`F_\delta = F_\delta(\hat{\mu},
  \hat{\Sigma}, \hat{\lambda})`.

For each value of :math:`\delta` and :math:`h`:

- Generate :math:`\mathbf{X}_1, \ldots, \mathbf{X}_{k-1}  \sim F_0`, for 
  :math:`\delta = 0`;
- Generate :math:`\mathbf{X}_k \sim F_\delta`;
- Compute the :math:`k`-sample test statistic between :math:`\mathbf{X}_1, 
  \mathbf{X}_2, \ldots, \mathbf{X}_k` with kernel parameter :math:`h`;
- Compute the power of the test. If it is greater than 0.5, 
  select :math:`h` as the optimal value.
- If an optimal value has not been selected, choose the :math:`h` which
  corresponds to maximum power.

The available `alternative` options are:

- **location** alternatives, :math:`F_\delta = 
  SN_d(\hat{\mu} + \delta, \hat{\Sigma}, \hat{\lambda})`, with 
  :math:`\delta = 0.2, 0.3, 0.4`;
- **scale** alternatives, 
  :math:`F_\delta = SN_d(\hat{\mu}, \hat{\Sigma} \cdot \delta, \hat{\lambda})`, 
  with :math:`\delta = 1.1, 1.3, 1.5`;
- **skewness** alternatives, 
  :math:`F_\delta = SN_d(\hat{\mu}, \hat{\Sigma}, \hat{\lambda} + \delta)`, 
  with :math:`\delta = 0.2, 0.3, 0.6`.

The values of :math:`h = 0.6, 1, 1.4, 1.8, 2.2` and :math:`N = 50` are set as 
default values. The function `select_h()` allows the user to 
set the values of :math:`\delta` and :math:`h` for a more extensive grid search. 
We suggest a more extensive grid search when computational resources 
permit.

.. note::
   Please be aware that the ``select_h()`` function may take a significant 
   amount of time to run, especially with larger datasets or when using a 
   larger number of parameters in ``h_values`` and ``delta``. Consider 
   this when applying the function to large or complex data.

References
----------

Markatou, M., & Saraceno, G. (2024). A unified framework for multivariate two-sample and k-sample 
kernel-based quadratic distance goodness-of-fit tests. arXiv preprint arXiv:2407.16374.

Saraceno, G., Markatou, M., Mukhopadhyay, R., & Golzy, M. (2024). Goodness-of-Fit and Clustering of Spherical Data: the 
QuadratiK package in R and Python. arXiv preprint arXiv:2402.02290.
