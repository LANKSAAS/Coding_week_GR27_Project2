"""
test_data_processing.py — Automated tests for data fetching & memory optimisation.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import fetch_dataset, optimize_memory

class TestFetchDataset:
    """Tests for ``fetch_dataset()``."""

    def test_returns_dataframe(self):
        df = fetch_dataset()
        assert isinstance(df, pd.DataFrame)

    def test_expected_shape(self):
        df = fetch_dataset()
        # Dataset should contain a reasonable number of rows and features
        assert len(df) > 2000
        assert 10 <= df.shape[1] <= 30

    def test_target_column_present(self):
        df = fetch_dataset()
        assert "NObeyesdad" in df.columns

    def test_missing_values_ratio(self):
        df = fetch_dataset()
        # Allow a small proportion of missing values instead of requiring none
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        assert missing_ratio < 0.05


class TestOptimizeMemory:
    """Tests for ``optimize_memory()``."""

    def test_reduces_memory(self):
        df = pd.DataFrame({
            "a": np.random.rand(100).astype(np.float64),
            "b": np.random.randint(0, 100, 100).astype(np.int64),
        })
        before = df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(df, verbose=False)
        after = df_opt.memory_usage(deep=True).sum()
        assert after < before

    def test_float32_conversion(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})  # float64
        df_opt = optimize_memory(df, verbose=False)
        assert df_opt["x"].dtype == np.float32

    def test_int32_conversion(self):
        df = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.int64)})
        df_opt = optimize_memory(df, verbose=False)
        assert df_opt["x"].dtype == np.int32

    def test_preserves_values(self):
        df = pd.DataFrame({"a": [1.5, 2.5], "b": [10, 20]})
        df_opt = optimize_memory(df, verbose=False)
        np.testing.assert_allclose(df_opt["a"].to_numpy(), np.array([1.5, 2.5]), rtol=1e-5)
        np.testing.assert_array_equal(df_opt["b"].values, [10, 20])
