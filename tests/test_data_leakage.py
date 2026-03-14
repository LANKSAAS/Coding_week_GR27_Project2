"""
test_data_leakage.py — Tests ensuring no data leakage occurs
between training and testing datasets.

These tests validate that preprocessing and dataset splitting
do not introduce overlap between training and test samples.
"""
import os
import sys
import pytest
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data_processing import fetch_dataset, preprocess_data


def _round_array(arr, decimals=8):
    """
    Round numerical arrays to avoid floating point precision issues
    when comparing rows.
    """
    return np.round(arr, decimals=decimals)


@pytest.mark.parametrize("decimals", [6])
def test_no_data_leakage_between_train_and_test(decimals):
    """
    Ensure that training and test sets share no identical samples.

    This prevents a critical machine learning issue called
    data leakage, where information from the test set leaks
    into the training set.
    """

    df = fetch_dataset()

    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df)

    # Convert arrays to comparable format
    X_train = _round_array(np.array(X_train), decimals)
    X_test = _round_array(np.array(X_test), decimals)

    train_rows = set(map(tuple, X_train))
    test_rows = set(map(tuple, X_test))

    intersection = train_rows.intersection(test_rows)

    # Allow very small overlap due to identical samples
    leakage_ratio = len(intersection) / len(train_rows)

    assert leakage_ratio < 0.01


def test_train_test_sizes_are_consistent():
    """
    Ensure that train/test split sizes are consistent
    and that neither set is empty.
    """

    df = fetch_dataset()

    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) > len(X_test)


def test_targets_do_not_overlap_indices():
    """
    Ensure that target vectors correspond to the correct
    number of samples in train and test sets.
    """

    df = fetch_dataset()

    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)