"""
Central configuration for the Student Grade Predictor.

Keeping feature order, performance bands, validation ranges and file paths in
one place satisfies NFR-13 ("important parameters should be configurable") and
FR-12 ("maintain the expected feature order when generating predictions").

Every other module imports from here so there is a single source of truth.
"""

import os

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root so they work from any cwd)
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

DATA_PATH = os.path.join(DATA_DIR, "student_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "grade_prediction_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# --------------------------------------------------------------------------- #
# Machine-learning schema (SRS section 6)
# --------------------------------------------------------------------------- #
# The order of this list is the contract between training and prediction.
FEATURES = [
    "study_hours",
    "attendance",
    "assignment_marks",
    "previous_marks",
    "internal_marks",
]

TARGET = "final_score"

# Train/test split configuration (SRS section 6.4)
TEST_SIZE = 0.20
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Human-friendly labels for the UI (SRS section 15 - usability)
# --------------------------------------------------------------------------- #
FEATURE_LABELS = {
    "study_hours": "Study Hours per Day",
    "attendance": "Attendance (%)",
    "assignment_marks": "Assignment Marks",
    "previous_marks": "Previous Examination Marks",
    "internal_marks": "Internal Assessment Marks",
}

# --------------------------------------------------------------------------- #
# Valid input ranges for validation (SRS section 6.1, 14.2)
# Each entry: (minimum, maximum) - values outside the range are rejected.
# --------------------------------------------------------------------------- #
INPUT_RANGES = {
    "study_hours": (0.0, 15.0),
    "attendance": (0.0, 100.0),
    "assignment_marks": (0.0, 100.0),
    "previous_marks": (0.0, 100.0),
    "internal_marks": (0.0, 100.0),
}

# The target score is always on a 0-100 scale.
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# --------------------------------------------------------------------------- #
# Performance classification bands (SRS section 3.6)
# These are configurable project rules, NOT an official grading policy.
# Ordered from highest to lowest; the first band whose lower bound the score
# meets or exceeds wins.
# --------------------------------------------------------------------------- #
PERFORMANCE_BANDS = [
    (90.0, "Excellent"),
    (75.0, "Very Good"),
    (60.0, "Good"),
    (50.0, "Average"),
    (0.0, "Needs Improvement"),
]

ALGORITHM_NAME = "Linear Regression (Ordinary Least Squares)"
