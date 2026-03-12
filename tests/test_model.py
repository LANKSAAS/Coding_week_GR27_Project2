"""
test_model.py — Automated tests for model prediction.
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _artefacts_exist():
    return all(
        os.path.exists(os.path.join(DATA_DIR, f))
        for f in ("best_model.joblib", "label_encoders.joblib", "feature_columns.joblib")
    )


@pytest.mark.skipif(not _artefacts_exist(), reason="Model artefacts not found — run train_model.py first")
class TestModelPrediction:
    """Tests for the saved best model."""

    @pytest.fixture(autouse=True)
    def load_artefacts(self):
        self.model = joblib.load(os.path.join(DATA_DIR, "best_model.joblib"))
        self.label_encoders = joblib.load(os.path.join(DATA_DIR, "label_encoders.joblib"))
        self.feature_columns = joblib.load(os.path.join(DATA_DIR, "feature_columns.joblib"))

    def _sample_input(self):
        """Create a realistic encoded sample."""
        return pd.DataFrame(
            [np.zeros(len(self.feature_columns))],
            columns=self.feature_columns,
        )

    def test_predict_returns_single_value(self):
        X = self._sample_input()
        pred = self.model.predict(X)
        assert pred.shape == (1,)

    def test_predict_proba_shape(self):
        X = self._sample_input()
        proba = self.model.predict_proba(X)
        n_classes = len(self.label_encoders["target"].classes_)
        assert proba.shape == (1, n_classes)

    def test_probabilities_sum_to_one(self):
        X = self._sample_input()
        proba = self.model.predict_proba(X)
        np.testing.assert_almost_equal(proba.sum(), 1.0, decimal=5)

    def test_prediction_in_valid_range(self):
        X = self._sample_input()
        pred = self.model.predict(X)[0]
        n_classes = len(self.label_encoders["target"].classes_)
        assert 0 <= pred < n_classes

    def test_model_comparison_csv_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "model_comparison.csv"))
