"""
This script demonstrates how to select the optimal bandwidth for a two sample test using the QuadratiK library.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm

from QuadratiK.kernel_test import select_h

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


# Perform the algorithm for selecting h
h_selected, all_powers, plot = select_h(
    x=X_2, y=Y_2, alternative="location", power_plot=True
)
plt.show()
# plot.savefig('two_sample_power_plot.png')
print(f"Selected h is: {h_selected}")
