"""
train_model.py — Train & evaluate multiple classifiers on the Obesity dataset.

Models trained:
  1. Random Forest
  2. XGBoost
  3. LightGBM
  4. CatBoost

Evaluation metrics (weighted / OVR where applicable):
  • Accuracy, Precision, Recall, F1-score, ROC-AUC

Outputs saved to ``data/``:
  • best_model.joblib          — the highest-scoring model
  • label_encoders.joblib      — fitted LabelEncoders (features + target)
  • feature_columns.joblib     — ordered feature column list
  • model_comparison.csv       — metric comparison across all models
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Ensure the src package is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import fetch_dataset, optimize_memory, preprocess_data

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _evaluate(model, X_test, y_test, n_classes: int) -> dict:
    """Return a dict of evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)

    # Predict probabilities for ROC-AUC (OVR)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted",
            )
        except ValueError:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    return metrics



def train_and_evaluate():
    """Full training pipeline: fetch → preprocess → train → evaluate → save."""

    # 1. Fetch & optimise ------------------------------------------------
    print("=" * 60)
    print("  Obesity Risk Estimation — Model Training Pipeline")
    print("=" * 60)

    df = fetch_dataset()
    df = optimize_memory(df)

    X_train, X_test, y_train, y_test, label_encoders, feature_columns = (
        preprocess_data(df)
    )

    n_classes = len(np.unique(y_train))
    print(f"\n[INFO] Classes: {n_classes}  |  "
          f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # 2. Define models ---------------------------------------------------
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.1,
            auto_class_weights="Balanced",
            random_state=42,
            verbose=0,
        ),
    }

    # 3. Train & evaluate ------------------------------------------------
    results = {}
    best_score = -1
    best_name = None
    best_model = None

    for name, model in models.items():
        print(f"[TRAIN] {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test, n_classes)
        results[name] = metrics
        print(f"F1={metrics['F1-Score']:.4f}  Acc={metrics['Accuracy']:.4f}  "
              f"AUC={metrics['ROC-AUC']:.4f}")

        if metrics["F1-Score"] > best_score:
            best_score = metrics["F1-Score"]
            best_name = name
            best_model = model

    # 4. Save artefacts --------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(DATA_DIR, "best_model.joblib"))
    joblib.dump(label_encoders, os.path.join(DATA_DIR, "label_encoders.joblib"))
    joblib.dump(feature_columns, os.path.join(DATA_DIR, "feature_columns.joblib"))

    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"
    comparison_df.to_csv(os.path.join(DATA_DIR, "model_comparison.csv"))

    print(f"\n{'=' * 60}")
    print(f"  Best model: {best_name}  (F1 = {best_score:.4f})")
    print(f"{'=' * 60}")
    print(f"\n[SAVED] data/best_model.joblib")
    print(f"[SAVED] data/label_encoders.joblib")
    print(f"[SAVED] data/feature_columns.joblib")
    print(f"[SAVED] data/model_comparison.csv")

    print("\n" + comparison_df.to_string())

    return best_model, results, label_encoders, feature_columns


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_evaluate()
