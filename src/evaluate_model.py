"""
Model evaluation module (SRS Feature 4, FR-16, section 6.5).

Computes the standard regression metrics: MAE, MSE, RMSE and R2. The
`regression_metrics` function is reused by the training script and can also be
run standalone to re-evaluate a saved model against the dataset.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, MSE, RMSE and R2 as a dict of floats (SRS 6.5)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def format_metrics(metrics: dict) -> str:
    """Human-readable, one-per-line summary of the metric dict."""
    return (
        f"MAE  (Mean Absolute Error)     : {metrics['mae']:.3f}\n"
        f"MSE  (Mean Squared Error)      : {metrics['mse']:.3f}\n"
        f"RMSE (Root Mean Squared Error) : {metrics['rmse']:.3f}\n"
        f"R2   (Coefficient of Determ.)  : {metrics['r2']:.4f}"
    )


def main() -> None:
    """Standalone: load the saved model and re-evaluate it on the dataset."""
    from sklearn.model_selection import train_test_split

    from src import config, preprocessing
    from src.prediction import load_model

    bundle = load_model()
    model = bundle["model"]

    X, y, _ = preprocessing.prepare()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    metrics = regression_metrics(y_test, model.predict(X_test))
    print("Evaluation on held-out test set")
    print("-" * 40)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
