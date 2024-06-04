from ._utils import _qq_plot_twosample, _extract_3d, _qq_plot_onesample
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

plt.ioff()


def qq_plot(x, y=None, dist="norm"):
    """
    The function qq_plot is used to create a quantile-quantile plot,
    either for a single sample or for two samples.

    Parameters
    ----------
        x : numpy.ndarray
            The `x` parameter represents the data for which you want to
            create a QQ plot. It can be a single variable or an array-like
            object containing multiple variables

        y : numpy.ndarray, optional
            The parameter `y` is an optional argument that represents the second
            sample for a two-sample QQ plot. If provided, the function will generate
            a QQ plot comparing the two samples

        dist : str, optional
            Supports all the scipy.stats.distributions. The `dist` parameter specifies
            the distribution to compare the data against in the QQ plot. By default,
            it is set to "norm" which represents the normal distribution. However, you can
            specify a different distribution if you want to compare the data against
            a different distribution. Defaults to "norm".

    Returns
    -------
        Returns QQ plots.

    Examples
    --------
    >>> import numpy as np
    >>> from QuadratiK.tools import qq_plot
    >>> np.random.seed(42)
    >>> X = np.random.randn(100,4)
    >>> qq_plot(X)
    """

    if y is None:
        if dist is not None:
            return _qq_plot_onesample(x, dist)
    else:
        return _qq_plot_twosample(x, y)


def sphere3d(x, y=None):
    """
    The function sphere3d creates a 3D scatter plot with a sphere
    as the surface and data points plotted on it.

    Parameters
    ----------
        x : numpy.ndarray, pandas.DataFrame
            The parameter `x` represents the input data for the scatter plot.
            It should be a 2D array-like object with shape (n_samples, 3),
            where each row represents the coordinates of a point in
            3D space.

        y : numpy.ndarray, list, pandas.series, optional
            The parameter `y` is an optional input that determines the color and
            shape of each data point in the plot. If `y` is not provided, the
            scatter plot will have the default marker symbol and color.


    Returns
    -------
        Returns a 3D plot of a sphere with data points plotted on it.

    Examples
    --------
    >>> from QuadratiK.tools import sphere3d
    >>> np.random.seed(42)
    >>> X = np.random.randn(100,3)
    >>> sphere3d(X)
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if isinstance(y, pd.DataFrame):
        y = y.to_numpy().flatten()
    elif isinstance(y, pd.Series):
        y = y.values
    elif isinstance(y, np.ndarray):
        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y = y.flatten()

    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
    x1 = r * sin(phi) * cos(theta)
    y1 = r * sin(phi) * sin(theta)
    z1 = r * cos(phi)
    xx, yy, zz = _extract_3d(x)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x1,
            y=y1,
            z=z1,
            colorscale=[[0, "#DCDCDC"], [1, "#DCDCDC"]],
            opacity=0.5,
            showscale=False,
        )
    )
    if y is None:
        fig.add_trace(
            go.Scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="markers",
                marker=dict(size=5, colorscale="turbo", showscale=False),
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="markers",
                marker=dict(
                    size=5,
                    color=y,
                    colorscale="turbo",
                    showscale=False,
                ),
            )
        )

    fig.update_layout(
        title="",
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode="data",
        ),
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_clusters_2d(x, y=None):
    """
    This function plots a 2D scatter plot of data points,
    with an optional argument to color the points based on
    a cluster label, and also plots a unit circle.

    Parameters
    ----------
        x : numpy.ndarray, pandas.DataFrame
            The parameter `x` is a 2-dimensional array or matrix
            containing the coordinates of the data points to be plotted.
            Each row of `x` represents the coordinates of a single data point
            in the 2-dimensional space.

        y : numpy.ndarray, pandas.DataFrame, optional
            The parameter `y` is an optional array that represents the labels
            or cluster assignments for each data point in `x`.
            If `y` is provided, the data points will be colored according to their
            labels or cluster assignments.

    Returns
    -------
        A matplotlib figure object.

    Examples
    --------
    >>> import numpy as np
    >>> from QuadratiK.tools import plot_clusters_2d
    >>> np.random.seed(42)
    >>> X = np.random.randn(100,2)
    >>> X = X/np.linalg.norm(X,axis = 1, keepdims=True)
    >>> plot_clusters_2d(X)
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if isinstance(y, pd.DataFrame):
        y = y.to_numpy().flatten()
    elif isinstance(y, pd.Series):
        y = y.values
    elif isinstance(y, np.ndarray):
        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y = y.flatten()

    fig = plt.figure()
    if y is not None:
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", edgecolors="k")
    else:
        plt.scatter(x[:, 0], x[:, 1], edgecolors="k")

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)

    plt.plot(unit_circle_x, unit_circle_y, linestyle="dashed", color="red")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.close()
    return fig
