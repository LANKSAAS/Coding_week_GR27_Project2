"""
test_model.py — Automated tests for trained model artefacts.

These tests verify that the trained model and related artefacts
produced by train_model.py behave as expected.
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib


# Path to model artefacts
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _artefacts_exist():
    """
    Check whether required model artefacts exist.

    If artefacts are missing, tests will be skipped.
    """
    required_files = [
        "best_model.joblib",
        "label_encoder.joblib",
        "feature_columns.joblib",
        "model_comparison.csv",
    ]

    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in required_files)


@pytest.mark.skipif(
    not _artefacts_exist(),
    reason="Model artefacts not found — run train_model.py first"
)
class TestModelPrediction:
    """
    Tests validating predictions of the saved best model.
    """

    @pytest.fixture(autouse=True)
    def load_artefacts(self):
        """
        Load model artefacts before each test.
        """

        self.model = joblib.load(os.path.join(DATA_DIR, "best_model.joblib"))
        self.label_encoder = joblib.load(os.path.join(DATA_DIR, "label_encoder.joblib"))
        self.feature_columns = joblib.load(os.path.join(DATA_DIR, "feature_columns.joblib"))

    def _sample_input(self):
        """
        Create a synthetic input sample compatible with the trained model.

        Since preprocessing is already applied before training,
        the model expects numeric features with the same column names.
        """

        return pd.DataFrame(
            [np.zeros(len(self.feature_columns))],
            columns=self.feature_columns,
        )

    def test_predict_returns_single_value(self):
        """
        The model should return exactly one prediction
        for a single input sample.
        """

        X = self._sample_input()

        pred = self.model.predict(X)

        assert pred.shape == (1,)

    def test_predict_proba_shape(self):
        """
        The model should return probabilities for each class.
        """

        X = self._sample_input()

        proba = self.model.predict_proba(X)

        n_classes = len(self.label_encoder.classes_)

        assert proba.shape == (1, n_classes)

    def test_probabilities_sum_to_one(self):
        """
        Predicted probabilities must sum to 1.
        """

        X = self._sample_input()

        proba = self.model.predict_proba(X)

        np.testing.assert_almost_equal(proba.sum(), 1.0, decimal=5)

    def test_prediction_in_valid_range(self):
        """
        Predicted class index must be within valid range.
        """

        X = self._sample_input()

        pred = self.model.predict(X)[0]

        n_classes = len(self.label_encoder.classes_)

        assert 0 <= pred < n_classes

    def test_prediction_is_deterministic(self):
        """
        The model should produce identical predictions
        when called multiple times with the same input.
        """

        X = self._sample_input()

        pred1 = self.model.predict(X)
        pred2 = self.model.predict(X)

        assert np.array_equal(pred1, pred2)

    def test_model_rejects_wrong_feature_count(self):
        """
        The model should raise an error if the number of input features is incorrect.
        """

        wrong_input = pd.DataFrame([[0, 1, 2]])

        with pytest.raises(Exception):
            self.model.predict(wrong_input)

    def test_model_comparison_csv_exists(self):
        """
        The model comparison file generated during training should exist.
        """

        assert os.path.exists(os.path.join(DATA_DIR, "model_comparison.csv"))