"""
Synthetic dataset generator for the Student Grade Predictor.

The SRS references data/student_data.csv but ships no data, so this script
produces a realistic, fully synthetic dataset. Using synthetic records also
respects the privacy requirements (SRS 9.4, 14.4): no real student is involved.

The data is built around a latent "ability" factor so the features are
correlated with each other and with the final score in a broadly linear way.
This gives Linear Regression a genuine signal to learn while keeping some
noise, so the reported R2 is high but not a suspicious 1.0.

Usage:
    python generate_data.py                 # 500 rows -> data/student_data.csv
    python generate_data.py --rows 1000     # custom size
"""

import argparse
import os

import numpy as np
import pandas as pd

from src import config


def generate(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of synthetic student records."""
    rng = np.random.default_rng(seed)

    # Latent student ability (standard normal) drives every observable feature.
    ability = rng.normal(0.0, 1.0, n_rows)

    # ----- Input features (kept within the ranges declared in config) -------
    study_hours = np.clip(rng.normal(5.5 + 1.6 * ability, 1.8), 0, 15)
    attendance = np.clip(rng.normal(78 + 9 * ability, 8), 40, 100)
    previous_marks = np.clip(rng.normal(66 + 14 * ability, 6), 25, 100)
    assignment_marks = np.clip(
        0.55 * previous_marks + 1.8 * study_hours + rng.normal(6, 5, n_rows), 0, 100
    )
    internal_marks = np.clip(
        0.45 * previous_marks + 0.30 * attendance + rng.normal(4, 5, n_rows), 0, 100
    )

    # ----- Target: final score as a linear blend of the features + noise ----
    final_score = (
        -8.0
        + 1.6 * study_hours
        + 0.15 * attendance
        + 0.20 * assignment_marks
        + 0.30 * previous_marks
        + 0.20 * internal_marks
        + rng.normal(0, 3.0, n_rows)
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 1),
            "attendance": np.round(attendance).astype(int),
            "assignment_marks": np.round(assignment_marks).astype(int),
            "previous_marks": np.round(previous_marks).astype(int),
            "internal_marks": np.round(internal_marks).astype(int),
            "final_score": np.round(final_score).astype(int),
        }
    )
    # Guarantee the column order matches the ML schema (features..., target).
    return df[config.FEATURES + [config.TARGET]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic student data.")
    parser.add_argument("--rows", type=int, default=500, help="Number of records.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out", default=config.DATA_PATH, help="Output CSV path."
    )
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} records to {args.out}")
    print("\nPreview:")
    print(df.head().to_string(index=False))
    print("\nSummary statistics:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
