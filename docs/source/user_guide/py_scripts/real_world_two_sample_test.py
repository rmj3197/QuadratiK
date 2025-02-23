"""
This script demonstrates how to perform a two sample test on the Wisconsin Breast Cancer dataset using the QuadratiK library.
"""

from QuadratiK.datasets import load_wisconsin_breast_cancer_data
from QuadratiK.kernel_test import KernelTest, select_h

X, y = load_wisconsin_breast_cancer_data(return_X_y=True, scaled=True)

# Create masks for Malignant (M) and Benign (B) tumors
malignant_mask = y == 1
benign_mask = y == 0

# Create X1 and X2 using the masks
X1 = X[malignant_mask.all(axis=1)]
X2 = X[benign_mask.all(axis=1)]

# Perform the algorithm for selecting h
h_selected, all_powers = select_h(
    x=X1, y=X2, alternative="skewness", method="subsampling", b=0.5, n_jobs=-1
)
print(f"Selected h is: {h_selected}")

# performing the two sample test
two_sample_test = KernelTest(h=h_selected, num_iter=150, random_state=42).test(X1, X2)

# printing two sample test object
print(two_sample_test)
