"""
This script demonstrates how to perform a k-sample test using the QuadratiK library.
"""

import numpy as np

from QuadratiK.kernel_test import KernelTest

np.random.seed(0)


size = 200
eps = 1
x1 = np.random.multivariate_normal(
    mean=[0, np.sqrt(3) * eps / 3], cov=np.eye(2), size=size
)
x2 = np.random.multivariate_normal(
    mean=[-eps / 2, -np.sqrt(3) * eps / 6], cov=np.eye(2), size=size
)
x3 = np.random.multivariate_normal(
    mean=[eps / 2, -np.sqrt(3) * eps / 6], cov=np.eye(2), size=size
)
# Merge the three samples into a single dataset
X_k = np.concatenate([x1, x2, x3])
# The memberships are needed for k-sample test
y_k = np.repeat(np.array([1, 2, 3]), size).reshape(-1, 1)

# performing the k-sample test
k_sample_test = KernelTest(h=1.5, method="subsampling", random_state=42).test(X_k, y_k)

# printing the summary for the k-sample test
print(k_sample_test.summary())
