"""
This script demonstrates how to perform a normality test using the QuadratiK library.
"""

import numpy as np

from QuadratiK.kernel_test import KernelTest

np.random.seed(78990)


# data generation
data_norm = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=500)

# performing the normality test
normality_test = KernelTest(
    h=0.4, num_iter=150, method="subsampling", random_state=42
).test(data_norm)

# printing the summary for normality test
print(normality_test.summary())
