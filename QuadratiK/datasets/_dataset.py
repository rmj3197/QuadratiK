from importlib import resources
from typing import Tuple, Union

import numpy as np
import pandas as pd


def load_wireless_data(
    desc: bool = False,
    return_X_y: bool = False,
    as_dataframe: bool = True,
    scaled: bool = False,
) -> Union[
    Tuple[str, pd.DataFrame, pd.DataFrame],
    Tuple[str, pd.DataFrame],
    Tuple[str, np.ndarray],
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray],
    pd.DataFrame,
    np.ndarray,
]:
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
        Determines whether the function should return the data as a pandas DataFrame (True)
        or as a numpy array (False). Defaults to True.

    scaled : boolean, optional
        Determines whether or not the data should be scaled. If set to True, the data will be
        divided by its Euclidean norm along each row. Defaults to False.

    Returns
    -------
    - If `desc=True`, `return_X_y=True`, `as_dataframe=True`:
        Returns a tuple containing: (str, pd.DataFrame, pd.DataFrame)

        - `fdescr` : str
            The description of the dataset.
        - `X` : pd.DataFrame
            A DataFrame with the features.
        - `y` : pd.DataFrame
            A DataFrame with the class labels.

    - If `desc=True`, `return_X_y=True`, `as_dataframe=False`:
        Returns a tuple containing: (str, np.ndarray, np.ndarray)

        - `fdescr` : str
            The description of the dataset.
        - `X` : np.ndarray
            A numpy array with the features .
        - `y` : np.ndarray
            A numpy array with the class labels .

    - If `desc=True`, `return_X_y=False`, `as_dataframe=True`:
        Returns a tuple containing: (str, pd.DataFrame)

        - `fdescr` : str
            The description of the dataset.
        - `data_df` : pd.DataFrame
            A DataFrame containing the entire dataset.

    - If `desc=True`, `return_X_y=False`, `as_dataframe=False`:
        Returns a tuple containing: (str, np.ndarray)

        - `fdescr` : str
            The description of the dataset.
        - `data` : np.ndarray
            A numpy array containing the entire dataset.

    - If `desc=False`, `return_X_y=True`, `as_dataframe=True`:
        Returns a tuple containing: (pd.DataFrame, pd.DataFrame)

        - `X` : pd.DataFrame
            A DataFrame with the features.
        - `y` : pd.DataFrame
            A DataFrame with the class labels.

    - If `desc=False`, `return_X_y=True`, `as_dataframe=False`:
        Returns a tuple containing: (np.ndarray, np.ndarray)

        - `X` : np.ndarray
            A numpy array with the features.
        - `y` : np.ndarray
            A numpy array with the class labels.

    - If `desc=False`, `return_X_y=False`, `as_dataframe=True`:
        Returns: pd.DataFrame

        - `data_df` : pd.DataFrame
            A DataFrame containing the entire dataset.

    - If `desc=False`, `return_X_y=False`, `as_dataframe=False`:
        Returns: np.ndarray

        - `data` : np.ndarray
            A numpy array containing the entire dataset.

    References
    ----------
    Rohra, J.G., Perumal, B., Narayanan, S.J., Thakur, P., Bhatt, R.B. (2017).
    User Localization in an Indoor Environment Using Fuzzy Hybrid of Particle Swarm Optimization
    & Gravitational Search Algorithm with Neural Networks. In: Deep, K., et al. Proceedings of
    Sixth International Conference on Soft Computing for Problem Solving. Advances in Intelligent
    Systems and Computing, vol 546. Springer, Singapore. https://doi.org/10.1007/978-981-10-3322-3_27.

    Source
    -------
    Bhatt, R. (2017). Wireless Indoor Localization. UCI Machine Learning Repository.
    https://doi.org/10.24432/C51880.

    Examples
    --------
    >>> from QuadratiK.datasets import load_wireless_data
    >>> X, y = load_wireless_data(return_X_y=True)
    """

    data = np.loadtxt(
        str(
            resources.files("QuadratiK.datasets").joinpath("data/wifi_localization.txt")
        )
    )

    if scaled:
        data[:, :-1] = data[:, :-1] / np.linalg.norm(
            data[:, :-1], axis=1, keepdims=True
        )

    feature_names = ["WS1", "WS2", "WS3", "WS4", "WS5", "WS6", "WS7", "Class"]

    if desc:
        desc_file = resources.files("QuadratiK.datasets").joinpath(
            "data/wireless_localization_dataset.rst"
        )
        fdescr = desc_file.read_text()

    if return_X_y:
        X = data[:, :-1]
        y = data[:, -1].astype(int)

    if as_dataframe:
        data_df = pd.DataFrame(data, columns=feature_names)
        data_df["Class"] = data_df["Class"].astype(int)

    if desc and return_X_y and as_dataframe:
        return (
            fdescr,
            pd.DataFrame(X, columns=feature_names[:-1]),
            pd.DataFrame(y, columns=["Class"]),
        )

    if desc and return_X_y and not as_dataframe:
        return (fdescr, X, y)

    if desc and not return_X_y and as_dataframe:
        return (fdescr, data_df)

    if desc and not return_X_y and not as_dataframe:
        return (fdescr, data)

    if not desc and return_X_y and as_dataframe:
        return (
            pd.DataFrame(X, columns=feature_names[:-1]),
            pd.DataFrame(y, columns=["Class"]),
        )

    if not desc and return_X_y and not as_dataframe:
        return (X, y)

    if not desc and not return_X_y and as_dataframe:
        return data_df

    if not desc and not return_X_y and not as_dataframe:
        return data


def load_wisconsin_breast_cancer_data(
    desc: bool = False,
    return_X_y: bool = False,
    as_dataframe: bool = True,
    scaled: bool = False,
) -> Union[
    Tuple[str, pd.DataFrame, pd.DataFrame],
    Tuple[str, pd.DataFrame],
    Tuple[str, np.ndarray],
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray],
    pd.DataFrame,
    np.ndarray,
]:
    """
    The Wisconsin breast cancer dataset data frame has 569 rows and 31 columns. The first 30 variables
    report the features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
    They describe characteristics of the cell nuclei present in the image.
    The last column indicates the class labels (Benign = 0 or Malignant = 1).

    The function load_wisconsin_breast_cancer_data loads the Breast Cancer Wisconsin (Diagnostic).

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
        Determines whether the function should return the data as a pandas DataFrame (True)
        or as a numpy array (False). Defaults to True.

    scaled : boolean, optional
        Determines whether or not the data should be scaled. If set to True, the data will be
        divided by its Euclidean norm along each row. Defaults to False.

    Returns
    -------
    - If `desc=True`, `return_X_y=True`, `as_dataframe=True`:
        Returns a tuple containing: (str, pd.DataFrame, pd.DataFrame)

        - `fdescr` : str
            The description of the dataset.
        - `X` : pd.DataFrame
            A DataFrame with the features.
        - `y` : pd.DataFrame
            A DataFrame with the class labels.

    - If `desc=True`, `return_X_y=True`, `as_dataframe=False`:
        Returns a tuple containing: (str, np.ndarray, np.ndarray)

        - `fdescr` : str
            The description of the dataset.
        - `X` : np.ndarray
            A numpy array with the features .
        - `y` : np.ndarray
            A numpy array with the class labels .

    - If `desc=True`, `return_X_y=False`, `as_dataframe=True`:
        Returns a tuple containing: (str, pd.DataFrame)

        - `fdescr` : str
            The description of the dataset.
        - `data_df` : pd.DataFrame
            A DataFrame containing the entire dataset.

    - If `desc=True`, `return_X_y=False`, `as_dataframe=False`:
        Returns a tuple containing: (str, np.ndarray)

        - `fdescr` : str
            The description of the dataset.
        - `data` : np.ndarray
            A numpy array containing the entire dataset.

    - If `desc=False`, `return_X_y=True`, `as_dataframe=True`:
        Returns a tuple containing: (pd.DataFrame, pd.DataFrame)

        - `X` : pd.DataFrame
            A DataFrame with the features.
        - `y` : pd.DataFrame
            A DataFrame with the class labels.

    - If `desc=False`, `return_X_y=True`, `as_dataframe=False`:
        Returns a tuple containing: (np.ndarray, np.ndarray)

        - `X` : np.ndarray
            A numpy array with the features.
        - `y` : np.ndarray
            A numpy array with the class labels.

    - If `desc=False`, `return_X_y=False`, `as_dataframe=True`:
        Returns: pd.DataFrame

        - `data_df` : pd.DataFrame
            A DataFrame containing the entire dataset.

    - If `desc=False`, `return_X_y=False`, `as_dataframe=False`:
        Returns: np.ndarray

        - `data` : np.ndarray
            A numpy array containing the entire dataset.

    References
    ----------
    Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993, July).
    Nuclear feature extraction for breast tumor diagnosis.
    In Biomedical image processing and biomedical visualization (Vol. 1905, pp. 861-870). SPIE.

    Source
    -------
    Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset].
    UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

    Examples
    --------
    >>> from QuadratiK.datasets import load_wisconsin_breast_cancer_data
    >>> X, y = load_wisconsin_breast_cancer_data(return_X_y=True)
    """

    data = np.loadtxt(
        str(
            resources.files("QuadratiK.datasets").joinpath(
                "data/wisconsin_breast_cancer.txt"
            )
        )
    )

    if scaled:
        data[:, :-1] = data[:, :-1] / np.linalg.norm(
            data[:, :-1], axis=1, keepdims=True
        )

    feature_names = [
        "radius1",
        "texture1",
        "perimeter1",
        "area1",
        "smoothness1",
        "compactness1",
        "concavity1",
        "concave_points1",
        "symmetry1",
        "fractal_dimension1",
        "radius2",
        "texture2",
        "perimeter2",
        "area2",
        "smoothness2",
        "compactness2",
        "concavity2",
        "concave_points2",
        "symmetry2",
        "fractal_dimension2",
        "radius3",
        "texture3",
        "perimeter3",
        "area3",
        "smoothness3",
        "compactness3",
        "concavity3",
        "concave_points3",
        "symmetry3",
        "fractal_dimension3",
        "Class",
    ]

    if desc:
        desc_file = resources.files("QuadratiK.datasets").joinpath(
            "data/wisconsin_breast_cancer_dataset.rst"
        )
        fdescr = desc_file.read_text()

    if return_X_y:
        X = data[:, :-1]
        y = data[:, -1].astype(int)

    if as_dataframe:
        data_df = pd.DataFrame(data, columns=feature_names)
        data_df["Class"] = data_df["Class"].astype(int)

    if desc and return_X_y and as_dataframe:
        return (
            fdescr,
            pd.DataFrame(X, columns=feature_names[:-1]),
            pd.DataFrame(y, columns=["Class"]),
        )

    if desc and return_X_y and not as_dataframe:
        return (fdescr, X, y)

    if desc and not return_X_y and as_dataframe:
        return (fdescr, data_df)

    if desc and not return_X_y and not as_dataframe:
        return (fdescr, data)

    if not desc and return_X_y and as_dataframe:
        return (
            pd.DataFrame(X, columns=feature_names[:-1]),
            pd.DataFrame(y, columns=["Class"]),
        )

    if not desc and return_X_y and not as_dataframe:
        return (X, y)

    if not desc and not return_X_y and as_dataframe:
        return data_df

    if not desc and not return_X_y and not as_dataframe:
        return data


def load_wine_data(
    desc: bool = False,
    return_X_y: bool = False,
    as_dataframe: bool = True,
    scaled: bool = False,
) -> Union[
    Tuple[str, pd.DataFrame, pd.DataFrame],
    Tuple[str, pd.DataFrame],
    Tuple[str, np.ndarray],
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[np.ndarray, np.ndarray],
    pd.DataFrame,
    np.ndarray,
]:
    """
    The wine data frame has 178 rows and 14 columns. The first 13 variables
    report 13 constituents found in each of the three types of wines.
    The last column indicates the class labels (1,2 or 3).

    The function load_wine_data loads the Wine dataset.

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
        Determines whether the function should return the data as a pandas DataFrame (True)
        or as a numpy array (False). Defaults to True.

    scaled : boolean, optional
        Determines whether or not the data should be scaled. If set to True, the data will be
        divided by its Euclidean norm along each row. Defaults to False.

    Returns
    -------
    - If `desc=True`, `return_X_y=True`, `as_dataframe=True`:
        Returns a tuple containing: (str, pd.DataFrame, pd.DataFrame)

        - `fdescr` : str
            The description of the dataset.
        - `X` : pd.DataFrame
            A DataFrame with the features.
        - `y` : pd.DataFrame
            A DataFrame with the class labels.

    - If `desc=True`, `return_X_y=True`, `as_dataframe=False`:
        Returns a tuple containing: (str, np.ndarray, np.ndarray)

        - `fdescr` : str
            The description of the dataset.
        - `X` : np.ndarray
            A numpy array with the features .
        - `y` : np.ndarray
            A numpy array with the class labels .

    - If `desc=True`, `return_X_y=False`, `as_dataframe=True`:
        Returns a tuple containing: (str, pd.DataFrame)

        - `fdescr` : str
            The description of the dataset.
        - `data_df` : pd.DataFrame
            A DataFrame containing the entire dataset.

    - If `desc=True`, `return_X_y=False`, `as_dataframe=False`:
        Returns a tuple containing: (str, np.ndarray)

        - `fdescr` : str
            The description of the dataset.
        - `data` : np.ndarray
            A numpy array containing the entire dataset.

    - If `desc=False`, `return_X_y=True`, `as_dataframe=True`:
        Returns a tuple containing: (pd.DataFrame, pd.DataFrame)

        - `X` : pd.DataFrame
            A DataFrame with the features.
        - `y` : pd.DataFrame
            A DataFrame with the class labels.

    - If `desc=False`, `return_X_y=True`, `as_dataframe=False`:
        Returns a tuple containing: (np.ndarray, np.ndarray)

        - `X` : np.ndarray
            A numpy array with the features.
        - `y` : np.ndarray
            A numpy array with the class labels.

    - If `desc=False`, `return_X_y=False`, `as_dataframe=True`:
        Returns: pd.DataFrame

        - `data_df` : pd.DataFrame
            A DataFrame containing the entire dataset.

    - If `desc=False`, `return_X_y=False`, `as_dataframe=False`:
        Returns: np.ndarray

        - `data` : np.ndarray
            A numpy array containing the entire dataset.

    References
    ----------
    Aeberhard, S., Coomans, D., & De Vel, O. (1994). Comparative analysis of statistical pattern recognition
    methods in high dimensional settings. Pattern Recognition, 27(8), 1065-1077.

    Source
    -------
    Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.

    Examples
    --------
    >>> from QuadratiK.datasets import load_wine_data
    >>> X, y = load_wine_data(return_X_y=True)
    """

    data = np.loadtxt(
        str(resources.files("QuadratiK.datasets").joinpath("data/wine.txt"))
    )

    if scaled:
        data[:, :-1] = data[:, :-1] / np.linalg.norm(
            data[:, :-1], axis=1, keepdims=True
        )

    feature_names = [
        "Alcohol",
        "Malicacid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "0D280_0D315_of_diluted_wines",
        "Proline",
        "Class",
    ]

    if desc:
        desc_file = resources.files("QuadratiK.datasets").joinpath(
            "data/wine_dataset.rst"
        )
        fdescr = desc_file.read_text()

    if return_X_y:
        X = data[:, :-1]
        y = data[:, -1].astype(int)

    if as_dataframe:
        data_df = pd.DataFrame(data, columns=feature_names)
        data_df["Class"] = data_df["Class"].astype(int)

    if desc and return_X_y and as_dataframe:
        return (
            fdescr,
            pd.DataFrame(X, columns=feature_names[:-1]),
            pd.DataFrame(y, columns=["Class"]),
        )

    if desc and return_X_y and not as_dataframe:
        return (fdescr, X, y)

    if desc and not return_X_y and as_dataframe:
        return (fdescr, data_df)

    if desc and not return_X_y and not as_dataframe:
        return (fdescr, data)

    if not desc and return_X_y and as_dataframe:
        return (
            pd.DataFrame(X, columns=feature_names[:-1]),
            pd.DataFrame(y, columns=["Class"]),
        )

    if not desc and return_X_y and not as_dataframe:
        return (X, y)

    if not desc and not return_X_y and as_dataframe:
        return data_df

    if not desc and not return_X_y and not as_dataframe:
        return data
