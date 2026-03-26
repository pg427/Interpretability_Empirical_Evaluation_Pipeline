from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
import io
import zipfile
import urllib.request

def load_dataset(name: str,
    *,
    higgs_sample_size: int | None = None,
    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    '''
    This function loads the dataset from sklearn.datasets and returns X, y and feature names

    :param name: Name of Dataset in sklearn.datasets (e.g., iris, wine)
    :return: Tabular Dataset
        X:  Array of features
        y: Array of labels
        feature_names: List of feature names
    '''

    if name == "iris":
        ds = load_iris()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = list(ds.feature_names)

    elif name == "wine":
        ds = load_wine()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = list(ds.feature_names)

    elif name == "breast_cancer":
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = list(ds.feature_names)

    elif name == "german_credit":
        url = "https://archive.ics.uci.edu/static/public/522/south%2Bgerman%2Bcredit.zip"

        with urllib.request.urlopen(url) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with zf.open("SouthGermanCredit.asc") as f:
                df = pd.read_csv(f, sep=r"\s+")

        X_df = df.drop(columns=["kredit"]).copy()
        y = df["kredit"].to_numpy()

        unique_vals = np.sort(pd.unique(y))
        if len(unique_vals) != 2:
            raise ValueError(f"Expected binary target for german_credit, got labels: {unique_vals}")

        if not np.array_equal(unique_vals, np.array([0, 1])):
            y = (y == unique_vals.max()).astype(int)

        X_df = pd.get_dummies(X_df, drop_first=True)

        X = X_df.to_numpy(dtype=np.float32)
        y = y.astype(int)
        feature_names = [str(c) for c in X_df.columns]

    elif name == "arcene":
        # OpenML ARCENE
        ds = fetch_openml(name="arcene", version=1, as_frame=True)

        X_df = ds.data.copy()
        y_series = pd.Series(ds.target).copy()

        # ARCENE labels are often strings like '1' and '-1'
        X = X_df.to_numpy(dtype=np.float32)
        y = y_series.astype(int).to_numpy()

        # force binary labels to 0/1
        unique_vals = np.unique(y)
        if len(unique_vals) == 2:
            y = (y == unique_vals.max()).astype(int)

        feature_names = [str(c) for c in X_df.columns]

    elif name == "isolet":
        # OpenML ISOLET
        ds = fetch_openml(name="isolet", version=1, as_frame=True)

        X_df = ds.data.copy()
        y_series = pd.Series(ds.target).copy()

        X = X_df.to_numpy(dtype=np.float32)

        # ISOLET target may come as strings; convert to integer labels
        y = y_series.astype(int).to_numpy()

        # if labels are 1..26 and you want 0..25:
        if y.min() == 1:
            y = y - 1

        feature_names = [str(c) for c in X_df.columns]

    elif name == "higgs":
        # UCI HIGGS
        dataset = fetch_ucirepo(id=280)
        X_df = dataset.data.features.copy()
        y_df = dataset.data.targets.copy()

        # convert target
        y = pd.Series(y_df.iloc[:, 0]).astype(int).to_numpy()

        # optional subsampling because full HIGGS is massive
        if higgs_sample_size is not None:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X_df), size=higgs_sample_size, replace=False)
            X_df = X_df.iloc[idx].reset_index(drop=True)
            y = y[idx]

        X = X_df.to_numpy(dtype=np.float32)
        feature_names = [str(c) for c in X_df.columns]

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return X, y, feature_names

def stratified_5fold_standardize(X: np.ndarray, y: np.ndarray, standardize: bool = True, shuffle: bool = True, random_state: int = 42):
    '''
    This function creates a 5fold cross validation for passed dataset (X,y)
    If standardize flag is set then standardizes the dataset as well.

    :param X: Array of features
    :param y: Array of labels
    :param standardize: Flag for standardization
    :param shuffle: Flag for shuffling dataset
    :param random_state: Random seed value for shuffling

    :return: Dictionary of folds. Each fold contains:
        fold: Fold No.
        train_idx: Indices of training samples
        test_idx: Indices of testing samples
        X_train: Features of training samples
        X_test: Features of testing samples
        y_train: Labels of training samples
        y_test: Labels of testing samples
        scaler: Standard Scaler object for standardization
    '''

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples,). Got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    #### CREATION 5-FOLDS ########
    splitter = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=random_state)
    folds: List[Dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train = X[train_idx].astype(np.float32)
        X_test = X[test_idx].astype(np.float32)
        y_train = y[train_idx].astype(int)
        y_test = y[test_idx].astype(int)

    ########## OPTIONAL : STANDARD TRANSFORMATION OF TRAINING AND TEST DATA #########
        scaler: Optional[MinMaxScaler] = None
        if standardize:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_test = scaler.transform(X_test).astype(np.float32)

        folds.append(
            {
                "fold": fold_i,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "scaler": scaler
            }
        )

    return folds


if __name__ == "__main__":
    '''
    IRIS Dataset Sample (w/ standardization)
    '''
    X, y, feature_names = load_dataset("iris")
    print(X[0], y[0], feature_names)

    folds_stand = stratified_5fold_standardize(X, y, standardize=True)
    print(folds_stand[0]["X_train"][0])

    folds_non_stand = stratified_5fold_standardize(X, y, standardize=False)
    print(folds_non_stand[0]["X_train"][0])

