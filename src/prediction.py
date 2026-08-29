"""
Prediction module (SRS Features 5-6-8, FR-18 .. FR-24, section 21.8).

Provides:
    * load_model            - load the trained model bundle (FR-18, NFR-09).
    * validate_input        - reject out-of-range / non-numeric input (FR-07, FR-24).
    * predict_score         - produce the numeric prediction (FR-21).
    * classify_performance  - map a score to a category (FR-23, SRS 3.6).
    * explain_prediction    - per-feature contributions (SRS 21.8 explainability).
"""

from __future__ import annotations

import os

import joblib
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# Model loading (FR-18, NFR-09, TC-07, TC-08)
# --------------------------------------------------------------------------- #
def load_model(path: str = None) -> dict:
    """Load the trained model bundle.

    Raises FileNotFoundError with an actionable message when the model file is
    absent, so the caller can show a meaningful error instead of crashing.
    """
    path = path or config.MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. "
            f"Train it first with: python -m src.train_model"
        )
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Input validation (FR-07, FR-24, section 14.2, TC-03, TC-04)
# --------------------------------------------------------------------------- #
def validate_input(values: dict) -> tuple[bool, list[str]]:
    """Validate a dict of {feature: value}.

    Returns (is_valid, errors). Every feature must be present, numeric and
    within its configured range. Examples of rejected input from the SRS:
    attendance = 150, attendance = -20, study_hours = -5.
    """
    errors: list[str] = []

    for feature in config.FEATURES:
        label = config.FEATURE_LABELS[feature]
        if feature not in values or values[feature] is None:
            errors.append(f"{label} is required.")
            continue

        try:
            value = float(values[feature])
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue

        low, high = config.INPUT_RANGES[feature]
        if value < low or value > high:
            errors.append(
                f"{label} must be between {low:g} and {high:g} (got {value:g})."
            )

    return (len(errors) == 0, errors)


# --------------------------------------------------------------------------- #
# Prediction (FR-20, FR-21)
# --------------------------------------------------------------------------- #
def _row_from_values(values: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with columns in the canonical order (FR-12)."""
    ordered = {f: [float(values[f])] for f in config.FEATURES}
    return pd.DataFrame(ordered, columns=config.FEATURES)


def predict_score(bundle: dict, values: dict) -> float:
    """Return the predicted final score, clipped to the valid 0-100 range."""
    model = bundle["model"]
    row = _row_from_values(values)
    raw = float(model.predict(row)[0])
    return max(config.SCORE_MIN, min(config.SCORE_MAX, raw))


# --------------------------------------------------------------------------- #
# Performance classification (FR-23, SRS 3.6)
# --------------------------------------------------------------------------- #
def classify_performance(score: float) -> str:
    """Map a numeric score to a performance category using config bands."""
    for lower_bound, label in config.PERFORMANCE_BANDS:
        if score >= lower_bound:
            return label
    return config.PERFORMANCE_BANDS[-1][1]


# --------------------------------------------------------------------------- #
# Explainability (SRS 21.8 - which factors drove this prediction)
# --------------------------------------------------------------------------- #
def explain_prediction(bundle: dict, values: dict) -> list[dict]:
    """Break the prediction into per-feature contributions.

    Contribution_i = coefficient_i * (value_i - training_mean_i).

    Centering on the training mean expresses each feature as how far it pushes
    THIS student above (+) or below (-) the typical student, which is far more
    interpretable than the raw coefficient*value term. The contributions are
    returned sorted by absolute impact (largest first).
    """
    coefficients = bundle["coefficients"]
    means = bundle.get("feature_means", {f: 0.0 for f in config.FEATURES})

    contributions = []
    for feature in config.FEATURES:
        value = float(values[feature])
        mean = float(means.get(feature, 0.0))
        coef = float(coefficients[feature])
        impact = coef * (value - mean)
        contributions.append(
            {
                "feature": feature,
                "label": config.FEATURE_LABELS[feature],
                "value": value,
                "mean": round(mean, 2),
                "impact": round(impact, 2),
                "direction": "positive" if impact >= 0 else "negative",
            }
        )

    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions


# --------------------------------------------------------------------------- #
# Convenience: full prediction in one call
# --------------------------------------------------------------------------- #
def predict(bundle: dict, values: dict) -> dict:
    """Validate, predict, classify and explain in a single call."""
    is_valid, errors = validate_input(values)
    if not is_valid:
        return {"ok": False, "errors": errors}

    score = predict_score(bundle, values)
    return {
        "ok": True,
        "score": round(score, 1),
        "performance": classify_performance(score),
        "explanation": explain_prediction(bundle, values),
    }
