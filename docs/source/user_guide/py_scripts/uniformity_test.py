"""
This script demonstrates how to perform a uniformity test using the QuadratiK library.
"""

import numpy as np

from QuadratiK.poisson_kernel_test import PoissonKernelTest

np.random.seed(0)


# data generation
z = np.random.normal(size=(200, 3))
data_unif = z / np.sqrt(np.sum(z**2, axis=1, keepdims=True))

# performing the uniformity test
unif_test = PoissonKernelTest(rho=0.7, random_state=42).test(data_unif)

# printing the summary for uniformity test
print(unif_test.summary())
