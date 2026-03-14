"""
data_processing.py
==================

Data ingestion and preprocessing utilities for the Obesity Risk Prediction project.

This module centralizes all operations related to dataset preparation:
    1. Downloading the dataset from the UCI Machine Learning Repository
    2. Caching the dataset locally
    3. Reducing DataFrame memory usage
    4. Building a robust preprocessing pipeline
    5. Splitting the dataset into training and testing sets

The preprocessing pipeline follows standard machine learning best practices:
    • Missing value imputation
    • Feature scaling for numerical variables
    • One-hot encoding for categorical variables

The resulting pipeline is compatible with scikit-learn models and can be
saved alongside trained models for reproducible inference.

Dataset:
    "Estimation of Obesity Levels Based on Eating Habits and Physical Condition"
"""

import os
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "obesity_data.csv")

UCI_DATASET_ID = 544

TARGET_COLUMN = "NObeyesdad"


# ---------------------------------------------------------------------
# Dataset retrieval
# ---------------------------------------------------------------------

def fetch_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Download or load the obesity dataset.

    This function retrieves the dataset from the UCI Machine Learning
    Repository using the `ucimlrepo` package. To avoid unnecessary
    network calls, the dataset is cached locally as a CSV file.

    Parameters
    ----------
    force_download : bool
        If True, the dataset is downloaded again even if cached locally.

    Returns
    -------
    pd.DataFrame
        Full dataset including features and target column.
    """

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if not force_download and os.path.exists(RAW_CSV_PATH):
        logger.info("Loading dataset from local cache")
        return pd.read_csv(RAW_CSV_PATH)

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "The 'ucimlrepo' package is required to download the dataset."
        )

    logger.info("Downloading dataset from UCI Machine Learning Repository")

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    df = pd.concat(
        [dataset.data.features, dataset.data.targets],
        axis=1
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' not found."
        )

    df.to_csv(RAW_CSV_PATH, index=False)

    logger.info(f"Dataset cached locally at {RAW_CSV_PATH}")

    return df


# ---------------------------------------------------------------------
# Memory optimization
# ---------------------------------------------------------------------

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by downcasting numeric types.

    Converts:
        float64 → float32
        int64   → int32
    """

    before = df.memory_usage(deep=True).sum() / 1024**2

    df_opt = df.copy()

    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = df_opt[col].astype(np.float32)

    for col in df_opt.select_dtypes(include=["int64"]).columns:
        df_opt[col] = df_opt[col].astype(np.int32)

    after = df_opt.memory_usage(deep=True).sum() / 1024**2

    logger.info(
        f"Memory usage reduced from {before:.2f}MB to {after:.2f}MB"
    )

    return df_opt


# ---------------------------------------------------------------------
# Preprocessing pipeline construction
# ---------------------------------------------------------------------

def build_preprocessing_pipeline(df: pd.DataFrame):
    """
    Create a scikit-learn preprocessing pipeline.

    Numerical features:
        • median imputation
        • standard scaling

    Categorical features:
        • most frequent imputation
        • one-hot encoding
    """

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if TARGET_COLUMN in categorical_cols:
        categorical_cols.remove(TARGET_COLUMN)

    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


# ---------------------------------------------------------------------
# Full preprocessing workflow
# ---------------------------------------------------------------------

def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Prepare dataset for machine learning training.

    Steps:
        1. Separate features and target
        2. Split dataset into train/test
        3. Fit preprocessing pipeline on training data
        4. Transform both train and test data

    Returns
    -------
    X_train : np.ndarray
    X_test : np.ndarray
    y_train : pd.Series
    y_test : pd.Series
    preprocessor : ColumnTransformer
    feature_names : list
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split BEFORE preprocessing (prevents data leakage)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessing_pipeline(df)

    X_train = preprocessor.fit_transform(X_train)

    X_test = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    df = fetch_dataset()

    df = optimize_memory(df)

    X_train, X_test, y_train, y_test, pipeline, feature_names = preprocess_data(df)

    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Number of features after preprocessing: {len(feature_names)}")