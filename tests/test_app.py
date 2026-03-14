"""
=========================================================
Streamlit Application Pipeline Tests (Optimized)
=========================================================

These tests validate the backend pipeline used by the
Streamlit dashboard without testing the UI itself.

The following components are verified:

1. Model loading
2. Dataset loading
3. Preprocessing pipeline
4. Model prediction capability
"""

import os
import sys
import joblib
import pytest
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.data_processing import fetch_dataset, preprocess_data


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture(scope="session")
def dataset():
    """Load dataset once."""
    return fetch_dataset()


@pytest.fixture(scope="session")
def preprocessing(dataset):
    """Run preprocessing once."""
    return preprocess_data(dataset)


@pytest.fixture(scope="session")
def model():
    """Load trained model once."""
    model_path = os.path.join(ROOT, "data", "best_model.joblib")
    assert os.path.exists(model_path), "Model file missing"
    return joblib.load(model_path)


# --------------------------------------------------
# TEST MODEL LOAD
# --------------------------------------------------

def test_model_loads(model):
    """Ensure the trained model loads correctly."""
    assert model is not None


# --------------------------------------------------
# TEST PREPROCESSING
# --------------------------------------------------

def test_preprocessing_outputs(preprocessing):

    X_train, X_test, y_train, y_test, transformer, feature_names = preprocessing

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(feature_names) > 0
    assert transformer is not None

def build_sample_dataframe(X_array, feature_names, index=0):
    """
    Convert numpy array sample into DataFrame with feature names.
    """
    sample = X_array[index].reshape(1, -1)
    return pd.DataFrame(sample, columns=feature_names)

# --------------------------------------------------
# TEST MODEL PREDICTION
# --------------------------------------------------

def test_model_prediction(model, preprocessing):

    X_train, X_test, y_train, y_test, transformer, feature_names = preprocessing

    sample = build_sample_dataframe(X_test, feature_names)

    prediction = model.predict(sample)

    assert prediction.shape[0] == 1


# --------------------------------------------------
# TEST MODEL PROBABILITIES
# --------------------------------------------------

def test_model_predict_proba(model, preprocessing):

    X_train, X_test, y_train, y_test, transformer, feature_names = preprocessing

    sample = build_sample_dataframe(X_test, feature_names)

    proba = model.predict_proba(sample)

    assert proba.shape[1] >= 2