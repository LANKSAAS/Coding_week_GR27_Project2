"""
data_processing.py — Data fetching, memory optimization, and preprocessing.

This module provides utilities to:
  1. Download the UCI Obesity dataset via the `ucimlrepo` package.
  2. Optimize DataFrame memory by downcasting numeric types.
  3. Preprocess data (encode categoricals, train/test split).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "obesity_data.csv")
UCI_DATASET_ID = 544


# ---------------------------------------------------------------------------
# 1. Dataset fetching
# ---------------------------------------------------------------------------
def fetch_dataset(force_download: bool = False) -> pd.DataFrame:
    """Download the UCI Obesity dataset and cache it locally.

    Uses the ``ucimlrepo`` package to fetch dataset **544** (Estimation of
    Obesity Levels Based on Eating Habits and Physical Condition).  The
    resulting DataFrame is saved to ``data/raw/obesity_data.csv`` so that
    subsequent calls load from disk instead of re-downloading.

    Parameters
    ----------
    force_download : bool, optional
        If *True*, re-download even when the local CSV already exists.

    Returns
    -------
    pd.DataFrame
        The full dataset with features **and** the target column
        ``NObeyesdad``.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if not force_download and os.path.exists(RAW_CSV_PATH):
        print(f"[INFO] Loading cached dataset from {RAW_CSV_PATH}")
        return pd.read_csv(RAW_CSV_PATH)

    print("[INFO] Downloading dataset from UCI ML Repository (id=544) …")
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    # Combine features and target into a single DataFrame
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"[INFO] Dataset saved to {RAW_CSV_PATH}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 2. Memory optimization
# ---------------------------------------------------------------------------
def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce DataFrame memory usage by downcasting numeric types.

    * ``float64`` → ``float32``
    * ``int64``   → ``int32``

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    verbose : bool, optional
        If *True*, print before/after memory statistics.

    Returns
    -------
    pd.DataFrame
        The optimized DataFrame (a copy).
    """
    before = df.memory_usage(deep=True).sum() / 1024**2  # MB

    df_opt = df.copy()
    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = df_opt[col].astype(np.float32)
    for col in df_opt.select_dtypes(include=["int64"]).columns:
        df_opt[col] = df_opt[col].astype(np.int32)

    after = df_opt.memory_usage(deep=True).sum() / 1024**2  # MB

    if verbose:
        print(f"[Memory] Before: {before:.4f} MB  ->  After: {after:.4f} MB  "
              f"(reduced {100 * (1 - after / before):.1f}%)")

    return df_opt


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "NObeyesdad",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Encode categorical features, split into train/test sets.

    All non-numeric columns (except the target) are label-encoded.  The
    target column is also label-encoded separately so that class names can
    be recovered later.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Train/test splits.
    label_encoders : dict[str, LabelEncoder]
        Mapping of column name → fitted ``LabelEncoder`` (includes the
        target under key ``"target"``).
    feature_columns : list[str]
        Ordered list of feature column names after encoding.
    """
    df = df.copy()
    label_encoders: dict[str, LabelEncoder] = {}

    # Encode target
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    label_encoders["target"] = le_target

    # Encode categorical features
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    feature_columns = [c for c in df.columns if c != target_col]
    X = df[feature_columns]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    return X_train, X_test, y_train, y_test, label_encoders, feature_columns


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = fetch_dataset()
    print(df.head())
    print(f"\nShape: {df.shape}")
    df_opt = optimize_memory(df)
