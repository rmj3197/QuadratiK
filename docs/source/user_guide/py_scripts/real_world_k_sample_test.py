"""
This script demonstrates how to perform a k-sample test on the Wine dataset using the QuadratiK library.
"""

from QuadratiK.datasets import load_wine_data
from QuadratiK.kernel_test import KernelTest, select_h

X, y = load_wine_data(return_X_y=True, scaled=True)

# Perform the algorithm for selecting h
h_selected, all_powers = select_h(
    x=X, y=y, alternative="skewness", n_jobs=-1, b=0.5, method="subsampling"
)
print(f"Selected h is: {h_selected}")

# performing the two sample test
k_sample_test = KernelTest(h=h_selected, num_iter=150, random_state=42).test(X, y)

# printing the summary for the two sample test
print(k_sample_test.summary())
