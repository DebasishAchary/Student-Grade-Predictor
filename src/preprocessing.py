"""
Data preprocessing module (SRS Feature 2, FR-08 .. FR-12).

Responsibilities:
    * Load the student dataset from CSV.
    * Report basic information (shape, columns, missing values).
    * Clean the data (coerce numerics, drop duplicates, handle missing values).
    * Separate the input features from the target variable, preserving the
      feature order declared in config.FEATURES.
"""

from __future__ import annotations

import pandas as pd

from src import config


def load_data(path: str = None) -> pd.DataFrame:
    """Load the dataset from a CSV file (FR-08).

    Raises FileNotFoundError with a clear message if the file is missing.
    """
    path = path or config.DATA_PATH
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            f"Run 'python generate_data.py' first to create it."
        ) from exc


def dataset_info(df: pd.DataFrame) -> dict:
    """Return a summary of the dataset (SRS section 28 - success metrics)."""
    return {
        "n_records": int(len(df)),
        "n_features": len(config.FEATURES),
        "columns": list(df.columns),
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset (FR-09, FR-10).

    Strategy (documented per SRS 9.3):
        * Keep only the expected columns.
        * Coerce every column to numeric; unparseable entries become NaN.
        * Drop exact duplicate rows.
        * Drop rows where the target is missing (cannot train/evaluate on them).
        * Impute missing *feature* values with that column's median, which is
          robust to outliers and never invents out-of-range values.
    """
    expected = config.FEATURES + [config.TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected columns: {expected}"
        )

    df = df[expected].copy()

    # Coerce to numeric (invalid strings -> NaN) so validation is meaningful.
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()

    # Rows without a target are unusable for supervised learning.
    df = df.dropna(subset=[config.TARGET])

    # Impute missing feature values with the column median.
    for col in config.FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    """Split into X (features, in canonical order) and y (target) - FR-11, FR-12."""
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y


def prepare(path: str = None):
    """Convenience pipeline: load -> clean -> split. Returns (X, y, info)."""
    df = load_data(path)
    info = dataset_info(df)
    clean = clean_data(df)
    X, y = split_features_target(clean)
    return X, y, info
