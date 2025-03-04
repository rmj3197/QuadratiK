"""
This script demonstrates how to select the optimal bandwidth for a k-sample test using the QuadratiK library.
"""

import numpy as np

from QuadratiK.kernel_test import select_h

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


# Perform the algorithm for selecting h
h_selected, all_powers = select_h(
    x=X_k, y=y_k, alternative="skewness", power_plot=False, method="subsampling", b=0.2
)
print(f"Selected h is: {h_selected}")
