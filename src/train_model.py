"""
Model training script (SRS Feature 3, FR-13, FR-14, FR-17, section 6).

Pipeline:
    load data -> clean -> split features/target -> train/test split ->
    fit LinearRegression -> evaluate -> persist model bundle + metrics.json

Run from the project root with either:
    python -m src.train_model
    python src/train_model.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# Allow "python src/train_model.py" by putting the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src import config, preprocessing
from src.evaluate_model import format_metrics, regression_metrics


def train():
    """Train the model, save it, and return (bundle, metrics)."""
    # 1. Load + clean + split into X / y ------------------------------------
    X, y, info = preprocessing.prepare()
    print(f"Loaded {info['n_records']} records "
          f"({info['total_missing']} missing values, "
          f"{info['duplicate_rows']} duplicates before cleaning).")

    # 2. Train / test split (FR-13, SRS 6.4) --------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    # 3. Fit the Linear Regression model (FR-14) ----------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Evaluate on the held-out test set (FR-15, FR-16) -------------------
    metrics = regression_metrics(y_test, model.predict(X_test))
    print("\nModel evaluation (held-out test set)")
    print("-" * 40)
    print(format_metrics(metrics))

    # 5. Persist the model bundle (FR-17) -----------------------------------
    bundle = {
        "model": model,
        "features": config.FEATURES,
        "target": config.TARGET,
        "algorithm": config.ALGORITHM_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "coefficients": dict(zip(config.FEATURES,
                                 [round(float(c), 4) for c in model.coef_])),
        "intercept": round(float(model.intercept_), 4),
        # Training-set feature means enable "vs. typical student" explanations.
        "feature_means": {f: round(float(X_train[f].mean()), 4)
                          for f in config.FEATURES},
    }

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {k: bundle[k] for k in
             ("algorithm", "trained_at", "n_train", "n_test",
              "metrics", "coefficients", "intercept")},
            fh,
            indent=2,
        )

    print(f"\nModel saved to   : {config.MODEL_PATH}")
    print(f"Metrics saved to : {config.METRICS_PATH}")
    print("\nLearned equation:")
    print(_equation_string(bundle))
    return bundle, metrics


def _equation_string(bundle: dict) -> str:
    parts = [f"{bundle['intercept']}"]
    for feat, coef in bundle["coefficients"].items():
        sign = "+" if coef >= 0 else "-"
        parts.append(f"{sign} {abs(coef)} x {feat}")
    return "  predicted_score = " + " ".join(parts)


if __name__ == "__main__":
    train()
