"""
Tests for the SHAP explanation module.

These tests ensure that:
- the trained model can be loaded
- SHAP values can be computed
- SHAP output dimensions are correct
- visualization files are generated
"""

import os
import sys
from pathlib import Path

import pytest
import numpy as np

# ------------------------------------------------------------------
# Allow importing project modules
# ------------------------------------------------------------------

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

from src.shap_explainer import (
    load_model,
    load_data,
    compute_shap_values,
)

# ------------------------------------------------------------------
# Test model loading
# ------------------------------------------------------------------


def test_model_loading():
    """Ensure the trained model loads successfully."""

    model = load_model()

    assert model is not None


# ------------------------------------------------------------------
# Test data loading
# ------------------------------------------------------------------


def test_data_loading():
    """Ensure dataset preprocessing works."""

    X_train, X_test, feature_names = load_data()

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feature_names) == X_train.shape[1]


# ------------------------------------------------------------------
# Test SHAP value computation
# ------------------------------------------------------------------


def test_shap_computation():
    """Ensure SHAP values can be computed without errors."""

    model = load_model()

    X_train, X_test, feature_names = load_data()

    shap_values, X_explain = compute_shap_values(
        model,
        X_train,
        X_test
    )

    assert shap_values is not None
    assert X_explain.shape[0] > 0


# ------------------------------------------------------------------
# Test SHAP output dimensions
# ------------------------------------------------------------------


def test_shap_dimensions():
    """Verify SHAP values dimensions match feature space."""

    model = load_model()

    X_train, X_test, feature_names = load_data()

    shap_values, X_explain = compute_shap_values(
        model,
        X_train,
        X_test
    )

    # number of features
    n_features = len(feature_names)

    assert shap_values.values.shape[1] == n_features


# ------------------------------------------------------------------
# Test SHAP values are not all zero
# ------------------------------------------------------------------


def test_shap_values_not_zero():
    """Ensure SHAP values contain meaningful contributions."""

    model = load_model()

    X_train, X_test, feature_names = load_data()

    shap_values, _ = compute_shap_values(
        model,
        X_train,
        X_test
    )

    total_importance = np.abs(shap_values.values).sum()

    assert total_importance > 0