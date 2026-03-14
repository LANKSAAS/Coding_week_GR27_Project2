"""
train_model.py
==============

This script trains and evaluates multiple machine learning models
for the Obesity Risk Estimation project.

Pipeline steps
--------------
1. Load the dataset from the UCI repository
2. Optimize dataset memory usage
3. Preprocess the data using the project preprocessing pipeline
4. Train multiple classification models
5. Evaluate model performance
6. Use cross-validation to select the best model
7. Save the trained model and artifacts
8. Generate evaluation visualizations

Models evaluated
----------------
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Evaluation metrics
------------------
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- ROC-AUC (One-vs-Rest)

Artifacts saved
---------------
data/
    best_model.joblib
    preprocessing_pipeline.joblib
    feature_columns.joblib
    model_comparison.csv
    model_comparison.png
    confusion_matrix.png

Usage
-----
Run the script from the project root:

    python src/train_model.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Allow direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing import (
    fetch_dataset,
    optimize_memory,
    preprocess_data,
)

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# -------------------------------------------------------------------
# Model evaluation
# -------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using multiple classification metrics.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "F1-Score": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    if hasattr(model, "predict_proba"):

        y_proba = model.predict_proba(X_test)

        try:
            metrics["ROC-AUC"] = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    return metrics


# -------------------------------------------------------------------
# Training pipeline
# -------------------------------------------------------------------

def train_and_evaluate():
    """
    Execute the full machine learning training pipeline.
    """

    print("=" * 60)
    print("Obesity Risk Estimation — Model Training")
    print("=" * 60)

    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------

    df = fetch_dataset()

    df = optimize_memory(df)

    # --------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------

    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Features after preprocessing : {len(feature_names)}\n")

    # --------------------------------------------------------------
    # Models
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------

    results = {}
    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():

        print(f"[TRAINING] {name}")

        # Cross-validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
            error_score=np.nan
        )
        cv_score = np.nanmean(cv_scores)
        cv_std = np.nanstd(cv_scores)

        print(f"Cross-validation F1-score: {cv_score:.4f} ± {cv_std:.4f}")

        # Train model
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        results[name] = metrics

        print(
            f"F1={metrics['F1-Score']:.4f} | "
            f"Accuracy={metrics['Accuracy']:.4f} | "
            f"AUC={metrics['ROC-AUC']:.4f}"
        )

        # Select best model using cross-validation
        if cv_score > best_score:

            best_score = cv_score
            best_model = model
            best_name = name

    # --------------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------------

    os.makedirs(DATA_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(DATA_DIR, "best_model.joblib"))

    joblib.dump(
        preprocessor,
        os.path.join(DATA_DIR, "preprocessing_pipeline.joblib"),
    )

    joblib.dump(
        feature_names,
        os.path.join(DATA_DIR, "feature_columns.joblib"),
    )
    joblib.dump(
        label_encoder,
        os.path.join(DATA_DIR, "label_encoder.joblib"),
    )

    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"

    comparison_df.to_csv(
        os.path.join(DATA_DIR, "model_comparison.csv")
    )

    # --------------------------------------------------------------
    # Model comparison plot
    # --------------------------------------------------------------

    plt.figure(figsize=(10,6))

    comparison_df[["Accuracy","F1-Score","ROC-AUC"]].plot(
        kind="bar"
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(os.path.join(DATA_DIR, "model_comparison.png"))

    plt.close()

    # --------------------------------------------------------------
    # Confusion matrix for best model
    # --------------------------------------------------------------

    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(os.path.join(DATA_DIR, "confusion_matrix.png"))

    plt.close()

    print("\n" + "=" * 60)
    print(f"Best model selected: {best_name}")
    print(f"Cross-validation F1-score: {best_score:.4f}")
    print("=" * 60)

    print("\nSaved artifacts:")
    print("data/best_model.joblib")
    print("data/preprocessing_pipeline.joblib")
    print("data/feature_columns.joblib")
    print("data/model_comparison.csv")
    print("data/model_comparison.png")
    print("data/confusion_matrix.png")

    return best_model, comparison_df


# -------------------------------------------------------------------

if __name__ == "__main__":

    train_and_evaluate()