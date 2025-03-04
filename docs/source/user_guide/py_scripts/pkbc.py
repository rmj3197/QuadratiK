"""
This script demonstrates how to perform PKBC clustering on the Wireless dataset using the QuadratiK library.
"""

import warnings

import matplotlib.pyplot as plt
import plotly.io as pio

from QuadratiK.datasets import load_wireless_data
from QuadratiK.spherical_clustering import PKBC

warnings.filterwarnings("ignore")

X, y = load_wireless_data(return_X_y=True)
# number of clusters tried are from 2 to 10
pkbc = PKBC(num_clust=range(2, 11), random_state=42).fit(X)

validation_metrics, elbow_plots = pkbc.validation(y_true=y)
plt.show()
elbow_plots.savefig("elbow_plots.png")
print(validation_metrics.round(2))

print(pkbc.summary())


# For viewing the plot please set:
pio.renderers.default = "browser"

pkbc_clusters = pkbc.plot(num_clust=4, y_true=y)
pkbc_clusters.show()
