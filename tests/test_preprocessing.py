import pytest
import numpy as np
from src.data_loader import encode, feature_engineering
from src.preprocessing import split
from tests.test_data_loader import _make_df


def _pipeline(n=200):
    df = _make_df(n)
    df = encode(df)
    df = feature_engineering(df)
    return df


def test_split_shapes():
    df = _pipeline()
    X_train, X_test, y_train, y_test = split(df)
    total = len(y_train) + len(y_test)
    assert abs(len(y_test) / total - 0.2) < 0.05


def test_split_stratification():
    df = _pipeline()
    X_train, X_test, y_train, y_test = split(df)
    diff = abs(y_train.mean() - y_test.mean())
    assert diff < 0.05, f"Stratification incorrecte : diff={diff:.3f}"


def test_split_no_overlap():
    df = _pipeline()
    X_train, X_test, y_train, y_test = split(df)
    assert len(set(X_train.index) & set(X_test.index)) == 0


def test_feature_count_consistent():
    df = _pipeline()
    X_train, X_test, y_train, y_test = split(df)
    assert X_train.shape[1] == X_test.shape[1]


def test_no_nan_numeric():
    df = _pipeline()
    X_train, _, _, _ = split(df)
    num_nan = X_train.select_dtypes("number").isnull().sum().sum()
    assert num_nan == 0, f"{num_nan} NaN dans colonnes numériques"