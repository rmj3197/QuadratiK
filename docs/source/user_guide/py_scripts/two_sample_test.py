"""
This script demonstrates how to perform a two sample test using the QuadratiK library.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm

from QuadratiK.kernel_test import KernelTest
from QuadratiK.tools import qq_plot

np.random.seed(0)


# data generation
X_2 = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=200)
Y_2 = skewnorm.rvs(
    size=(200, 4),
    loc=np.zeros(4),
    scale=np.ones(4),
    a=np.repeat(0.5, 4),
    random_state=20,
)
# performing the two sample test
two_sample_test = KernelTest(h=2, num_iter=150, random_state=42).test(X_2, Y_2)

# printing the summary for the two sample test
print(two_sample_test.summary())


two_sample_qq_plot = qq_plot(X_2, Y_2)
plt.show()
# two_sample_qq_plot.savefig('two_sample_qq_plot.png')
