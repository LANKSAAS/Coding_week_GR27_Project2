"""
SHAP model explanation module.

This script loads the trained model and dataset,
computes SHAP values, and generates interpretability
visualizations for the machine learning model.

Generated figures:
    data/shap_figures/
        ├── shap_summary.png
        ├── shap_feature_importance.png
        └── shap_waterfall.png

To reduce runtime, only a subset of samples is used
for SHAP computation.
"""

import sys
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

# ---------------------------------------------------------------------
# Allow pytest / script execution without path issues
# ---------------------------------------------------------------------

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.data_processing import fetch_dataset, preprocess_data

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"

MODEL_PATH = DATA_DIR / "best_model.joblib"

SHAP_DIR = DATA_DIR / "shap_figures"

SHAP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

def load_model():

    logger.info("Loading trained model")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ---------------------------------------------------------------------
# Load and preprocess dataset
# ---------------------------------------------------------------------

def load_data():

    logger.info("Loading dataset")

    df = fetch_dataset()

    X_train, X_test, y_train, y_test, preprocessor, _ = preprocess_data(df)

    # Extract feature names from ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    return X_train_df, X_test_df, feature_names

# ---------------------------------------------------------------------
# Compute SHAP values (optimized)
# ---------------------------------------------------------------------

def compute_shap_values(model, X_train_df, X_test_df):

    logger.info("Computing SHAP values (TreeExplainer optimized)")

    # Smaller subset for explanation
    sample_size = min(120, len(X_test_df))

    X_explain = X_test_df.sample(sample_size, random_state=42)

    # TreeExplainer optimized for tree-based models
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_explain)

    return shap_values, X_explain


# ---------------------------------------------------------------------
# SHAP summary plot
# ---------------------------------------------------------------------

def plot_summary(shap_values, X_explain_df):

    logger.info("Generating SHAP summary plot")

    shap.summary_plot(
        shap_values,
        X_explain_df,
        show=False
    )

    plt.tight_layout()

    path = SHAP_DIR / "shap_summary.png"

    plt.savefig(path, dpi=300)

    logger.info(f"Saved summary plot → {path}")

    plt.close()


# ---------------------------------------------------------------------
# SHAP feature importance plot
# ---------------------------------------------------------------------

def plot_feature_importance(shap_values, X_explain_df):

    logger.info("Generating SHAP feature importance plot")

    shap.summary_plot(
        shap_values,
        X_explain_df,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    path = SHAP_DIR / "shap_feature_importance.png"

    plt.savefig(path, dpi=300)

    logger.info(f"Saved feature importance plot → {path}")

    plt.close()


# ---------------------------------------------------------------------
# SHAP waterfall plot (multi-class compatible)
# ---------------------------------------------------------------------

def plot_waterfall(shap_values, X_explain_df, model, index=0):

    logger.info("Generating SHAP waterfall plot")

    # Predict class of the selected sample
    prediction = model.predict(X_explain_df.iloc[[index]])[0]

    # Extract explanation for predicted class
    explanation = shap_values[index, :, prediction]

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.tight_layout()

    path = SHAP_DIR / "shap_waterfall.png"

    plt.savefig(path, dpi=300)

    logger.info(f"Saved waterfall plot → {path}")

    plt.close()


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():

    logger.info("Starting SHAP explanation pipeline")

    model = load_model()

    X_train_df, X_test_df, feature_names = load_data()

    shap_values, X_explain_df = compute_shap_values(
        model,
        X_train_df,
        X_test_df
    )

    plot_summary(
        shap_values,
        X_explain_df
    )

    plot_feature_importance(
        shap_values,
        X_explain_df
    )

    plot_waterfall(
        shap_values,
        X_explain_df,
        model
    )

    logger.info("SHAP explanation completed")


# ---------------------------------------------------------------------

if __name__ == "__main__":

    main()