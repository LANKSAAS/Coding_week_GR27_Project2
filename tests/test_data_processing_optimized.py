"""
test_data_processing.py
=======================

Automated tests for the data processing module.

This test suite validates the core functionalities of the
`data_processing.py` module used in the obesity prediction project.

Tested components
-----------------

1. Dataset retrieval
   - Dataset can be downloaded or loaded from cache
   - Returned object is a pandas DataFrame
   - Expected structure of the dataset

2. Memory optimisation
   - Numeric columns are downcast correctly
   - Memory consumption decreases
   - Data integrity is preserved after conversion

These tests ensure that the preprocessing layer remains stable,
since many downstream components depend on it:

    • model training
    • SHAP explainability
    • Streamlit inference interface
"""

import sys
import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Allow importing the project source directory
# ------------------------------------------------------------------

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data_processing import fetch_dataset, optimize_memory, preprocess_data


# ==================================================================
# Dataset fetching tests
# ==================================================================

class TestFetchDataset:
    """
    Tests related to dataset retrieval from the UCI repository
    or from the local cache.

    These tests validate that the dataset used by the ML pipeline
    has the expected structure and basic quality properties.
    """

    def test_returns_dataframe(self):
        """fetch_dataset() should return a pandas DataFrame."""
        df = fetch_dataset()
        assert isinstance(df, pd.DataFrame)

    def test_expected_shape(self):
        """
        The dataset should contain a reasonable number of rows
        and columns for the obesity dataset.
        """
        df = fetch_dataset()

        assert len(df) > 2000
        assert 10 <= df.shape[1] <= 30

    def test_target_column_present(self):
        """
        The target column used for classification must exist.
        """
        df = fetch_dataset()
        assert "NObeyesdad" in df.columns

    def test_missing_values_ratio(self):
        """
        The dataset should not contain excessive missing values.

        A small proportion is tolerated since imputation is part
        of the preprocessing pipeline.
        """
        df = fetch_dataset()

        missing_ratio = df.isnull().sum().sum() / (
            df.shape[0] * df.shape[1]
        )

        assert missing_ratio < 0.05


# ==================================================================
# Memory optimisation tests
# ==================================================================

class TestOptimizeMemory:
    """
    Tests validating the behaviour of the memory optimisation utility.

    The function should:

        • downcast float64 → float32
        • downcast int64 → int32
        • reduce overall memory footprint
        • preserve numerical values
    """

    def test_reduces_memory(self):
        """
        Memory usage after optimisation should be smaller.
        """

        df = pd.DataFrame({
            "a": np.random.rand(100).astype(np.float64),
            "b": np.random.randint(0, 100, 100).astype(np.int64),
        })

        before = df.memory_usage(deep=True).sum()

        df_opt = optimize_memory(df)

        after = df_opt.memory_usage(deep=True).sum()

        assert after < before

    def test_float32_conversion(self):
        """
        float64 columns should be converted to float32.
        """

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0]
        })

        df_opt = optimize_memory(df)

        assert df_opt["x"].dtype == np.float32

    def test_int32_conversion(self):
        """
        int64 columns should be converted to int32.
        """

        df = pd.DataFrame({
            "x": np.array([1, 2, 3], dtype=np.int64)
        })

        df_opt = optimize_memory(df)

        assert df_opt["x"].dtype == np.int32

    def test_preserves_values(self):
        """
        Numeric values must remain unchanged after conversion.
        """

        df = pd.DataFrame({
            "a": [1.5, 2.5],
            "b": [10, 20],
        })

        df_opt = optimize_memory(df)

        np.testing.assert_allclose(
            df_opt["a"].to_numpy(),
            np.array([1.5, 2.5]),
            rtol=1e-5,
        )

        np.testing.assert_array_equal(
            df_opt["b"].values,
            [10, 20],
        )

# ==================================================================
# Preprocessing pipeline tests
# ==================================================================

class TestPreprocessData:
    """Tests for the preprocess_data() pipeline."""

    def test_preprocess_returns_expected_outputs(self):
        """
        preprocess_data should return the expected six outputs.
        """

        df = fetch_dataset()

        result = preprocess_data(df)

        assert len(result) == 6

    def test_train_test_split(self):
        """
        Training set should be larger than test set
        according to the default split (80/20).
        """

        df = fetch_dataset()

        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

        assert len(X_train) > len(X_test)
        assert len(y_train) > len(y_test)

    def test_no_nan_after_preprocessing(self):
        """
        Preprocessed data should not contain NaN values
        because imputers are used in the pipeline.
        """

        df = fetch_dataset()

        X_train, X_test, _, _, _, _ = preprocess_data(df)

        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_feature_names_match_output(self):
        """
        The number of feature names returned by the pipeline
        must match the number of columns in the transformed data.
        """

        df = fetch_dataset()

        X_train, X_test, _, _, _, feature_names = preprocess_data(df)

        assert X_train.shape[1] == len(feature_names)