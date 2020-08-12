from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from feature_extractor import fingerprint_features


def fetch_single_class_data(test_size: Union[float, int] = 0.2,
                            is_stratified: bool = True,
                            get_extracted_features: bool = True):
    """Split single label dataset into train and test.

    Parameters
    ----------
    test_size : float, int or None, optional (default=None)
        Should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.
    is_stratified : bool (default=True)
        Whether data splits in a stratified fashion or not.
    get_extracted_features : bool (default=True)
        Whether data splits are extracted feature or smiles.

    Returns
    -------

    """
    data = pd.read_csv('data/dataset_single.csv')
    # data = data[:100]

    if get_extracted_features:
        X = np.array([fingerprint_features(smile, size=2048)
                      for smile in data['smiles'].values])
        y = to_categorical(data['P1'].values)
    else:
        X = data['smiles'].values
        y = data['P1'].values

    # one_hot_y = to_categorical(data['P1'].values)
    # one_hot_y = data['P1'].values

    if is_stratified:
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=stratify)

    return X_train, X_test, y_train, y_test


def fetch_multi_class_data(test_size: Union[float, int] = 0.2):
    """Split single label dataset into train and test.

    Parameters
    ----------
    test_size : float, int or None, optional (default=None)
        Should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    Returns
    -------
    train and test datasets

    """
    data = pd.read_csv('data/dataset_multi.csv')
    X = data['smiles'].values
    y = data[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test
