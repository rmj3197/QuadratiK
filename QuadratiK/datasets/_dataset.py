import pandas as pd
import numpy as np
from importlib import resources


def load_wireless_data(desc=False, return_X_y=False, as_dataframe=True, scaled=False):
    """
    The wireless data frame has 2000 rows and 8 columns. The first 7 variables
    report the measurements of the Wi-Fi signal strength received from 7 Wi-Fi routers in an
    office location in Pittsburgh (USA). The last column indicates the class labels.

    The function load_wireless_data loads a wireless localization dataset.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------

        desc : boolean, optional 
            If set to `True`, the function will return the description along with the data. 
            If set to `False`, the description will not be included. Defaults to False.

        return_X_y : boolean, optional
            Determines whether the function should return the data as separate arrays (`X` and `y`). 
            Defaults to False.

        as_dataframe : boolean, optional
            Determines whether the function should return the data as a pandas DataFrame (Trues) 
            or as a numpy array (False). Defaults to True. 

        scaled : boolean, optional
            Determines whether or not the data should be scaled. If set to True, the data will be 
            divided by its Euclidean norm along each row. Defaults to False.

    Returns
    -------

        (data, target) : tuple, if return_X_y is True
            A tuple of two ndarray. The first containing a 2D array of shape
            (n_samples, n_features) with each row representing one sample and
            each column representing the features. The second ndarray of shape
            (n_samples,) containing the target samples.

        data : pandas.DataFrame, if as_dataframe is True
            Dataframe of the data with shape (n_samples, n_features + class)

        (desc, data, target) : tuple, if desc is True and return_X_y is True
            A tuple of description and two numpy.ndarray. The first containing a 2D 
            array of shape (n_samples, n_features) with each row representing 
            one sample and each column representing the features. The second 
            ndarray of shape (n_samples,) containing the target samples.

        (desc, data) : tuple, if desc is True and as_dataframe is True
            A tuple of description and pandas.DataFrame.
            Dataframe of the data with shape (n_samples, n_features + class)

    References
    ----------
        Rohra, J.G., Perumal, B., Narayanan, S.J., Thakur, P., Bhatt, R.B. (2017). 
        User Localization in an Indoor Environment Using Fuzzy Hybrid of Particle Swarm Optimization 
        & Gravitational Search Algorithm with Neural Networks. In: Deep, K., et al. Proceedings of 
        Sixth International Conference on Soft Computing for Problem Solving. Advances in Intelligent 
        Systems and Computing, vol 546. Springer, Singapore. https://doi.org/10.1007/978-981-10-3322-3_27

    Source
    -------
        Bhatt,Rajen. (2017). Wireless Indoor Localization. UCI Machine Learning Repository.
        https://doi.org/10.24432/C51880.

    Examples
    --------
    >>> from QuadratiK.datasets import load_wireless_data
    >>> X, y = load_wireless_data(return_X_y=True)
    """

    data = np.loadtxt(str(resources.files(
        "QuadratiK.datasets").joinpath("data/wifi_localization.txt")))

    if scaled:
        data = data/np.linalg.norm(data, axis=1, keepdims=True)

    feature_names = ["WS1", "WS2", "WS3", "WS4", "WS5", "WS6", "WS7", "Class"]

    if desc:
        desc_file = resources.files("QuadratiK.datasets").joinpath(
            "data/wireless_localization_dataset.rst")
        fdescr = desc_file.read_text()

    if return_X_y:
        X = data[:, :-1]
        y = data[:, -1]
        if desc:
            return (fdescr, X, y)
        else:
            return (X, y)

    if as_dataframe:
        if desc:
            return (fdescr, pd.DataFrame(data, columns=feature_names))
        else:
            return pd.DataFrame(data, columns=feature_names)
