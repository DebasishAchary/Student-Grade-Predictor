"""
Test suite for the Student Grade Predictor.

Covers the SRS functional test cases where they can be checked automatically:
    TC-02  valid input produces a numeric prediction
    TC-03  attendance = 150 is rejected
    TC-04  negative study hours is rejected
    TC-05  prediction generation from valid information
    TC-06  performance classification (score 82 -> Very Good)

Plus unit tests for preprocessing and the explainability feature.

Run from the project root:
    pytest -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_data import generate
from src import config, preprocessing
from src.prediction import (
    classify_performance,
    explain_prediction,
    predict_score,
    validate_input,
)

VALID_INPUT = {
    "study_hours": 7,
    "attendance": 90,
    "assignment_marks": 85,
    "previous_marks": 80,
    "internal_marks": 82,
}


@pytest.fixture(scope="module")
def bundle():
    """Train a small in-memory model so tests don't depend on the saved .pkl."""
    df = generate(n_rows=300, seed=1)
    X, y = preprocessing.split_features_target(df)
    model = LinearRegression().fit(X, y)
    return {
        "model": model,
        "features": config.FEATURES,
        "target": config.TARGET,
        "coefficients": dict(zip(config.FEATURES,
                                 [float(c) for c in model.coef_])),
        "intercept": float(model.intercept_),
        "feature_means": {f: float(X[f].mean()) for f in config.FEATURES},
    }


# --------------------------- classification (TC-06) ------------------------ #
@pytest.mark.parametrize(
    "score, expected",
    [
        (95, "Excellent"),
        (90, "Excellent"),
        (82, "Very Good"),   # TC-06
        (75, "Very Good"),
        (60, "Good"),
        (55, "Average"),
        (50, "Average"),
        (49, "Needs Improvement"),
        (0, "Needs Improvement"),
    ],
)
def test_classify_performance(score, expected):
    assert classify_performance(score) == expected


# --------------------------- input validation ----------------------------- #
def test_valid_input_passes():
    is_valid, errors = validate_input(VALID_INPUT)
    assert is_valid
    assert errors == []


def test_attendance_too_high_rejected():          # TC-03
    bad = dict(VALID_INPUT, attendance=150)
    is_valid, errors = validate_input(bad)
    assert not is_valid
    assert any("Attendance" in e for e in errors)


def test_negative_study_hours_rejected():         # TC-04
    bad = dict(VALID_INPUT, study_hours=-5)
    is_valid, errors = validate_input(bad)
    assert not is_valid
    assert any("Study Hours" in e for e in errors)


def test_non_numeric_rejected():
    bad = dict(VALID_INPUT, assignment_marks="abc")
    is_valid, errors = validate_input(bad)
    assert not is_valid


def test_missing_field_rejected():
    bad = {k: v for k, v in VALID_INPUT.items() if k != "internal_marks"}
    is_valid, errors = validate_input(bad)
    assert not is_valid


# --------------------------- prediction (TC-02, TC-05) --------------------- #
def test_prediction_is_numeric_in_range(bundle):
    score = predict_score(bundle, VALID_INPUT)
    assert isinstance(score, float)
    assert config.SCORE_MIN <= score <= config.SCORE_MAX


def test_prediction_responds_to_better_inputs(bundle):
    """A stronger student should score at least as high as a weaker one."""
    weak = {"study_hours": 1, "attendance": 50, "assignment_marks": 40,
            "previous_marks": 45, "internal_marks": 42}
    strong = {"study_hours": 12, "attendance": 98, "assignment_marks": 95,
              "previous_marks": 92, "internal_marks": 94}
    assert predict_score(bundle, strong) > predict_score(bundle, weak)


# --------------------------- explainability (SRS 21.8) --------------------- #
def test_explanation_shape_and_order(bundle):
    contributions = explain_prediction(bundle, VALID_INPUT)
    assert len(contributions) == len(config.FEATURES)
    impacts = [abs(c["impact"]) for c in contributions]
    assert impacts == sorted(impacts, reverse=True)  # sorted by |impact| desc


# --------------------------- preprocessing --------------------------------- #
def test_clean_data_handles_missing_and_duplicates():
    raw = pd.DataFrame(
        {
            "study_hours": [5, 5, np.nan, 8],
            "attendance": [85, 85, 70, 95],
            "assignment_marks": [78, 78, 60, 91],
            "previous_marks": [72, 72, 55, 89],
            "internal_marks": [75, 75, 58, 90],
            "final_score": [76, 76, 57, 93],
        }
    )
    clean = preprocessing.clean_data(raw)
    # One duplicate row removed -> 3 rows remain.
    assert len(clean) == 3
    # Missing study_hours was imputed, not left as NaN.
    assert clean["study_hours"].isna().sum() == 0


def test_split_preserves_feature_order():
    df = generate(n_rows=10, seed=2)
    X, y = preprocessing.split_features_target(df)
    assert list(X.columns) == config.FEATURES
    assert y.name == config.TARGET
