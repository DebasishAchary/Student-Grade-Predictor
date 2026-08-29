# 🎓 Student Grade Predictor

A machine-learning web application that predicts a student's expected **final
examination score** from five academic factors, using an Ordinary Least Squares
**Linear Regression** model. Built with Python, pandas, NumPy, scikit-learn and
Streamlit.

This project implements the *Student Grade Predictor* Software Requirements
Specification (SRS v1.0). It demonstrates a complete, end-to-end ML workflow —
data generation, preprocessing, training, evaluation, prediction, explainability
and an interactive dashboard.

> **Disclaimer:** predictions are estimates for educational purposes only. They
> are not guaranteed exam results and must not be the sole basis for any
> academic decision. See *Ethics & limitations* below.

---

## Features

- **Interactive dashboard** — enter student details and get an instant predicted
  score with a clear performance category.
- **Explainable predictions** — a per-factor breakdown showing which inputs push
  the student above or below a typical student (SRS §21.8).
- **Data insights** — scatter plots of each feature against the final score, plus
  an actual-vs-predicted plot.
- **Model transparency** — R², RMSE, MAE, MSE and the learned coefficients.
- **Input validation** — out-of-range or non-numeric input is rejected with clear
  messages.
- **Modular, tested code** — preprocessing, training, evaluation and prediction
  are separate modules with a pytest suite.

---

## Project structure

```
Student-Grade-Predictor/
├── app.py                 # Streamlit dashboard (UI)
├── generate_data.py       # Creates the synthetic dataset
├── requirements.txt
├── README.md
├── data/
│   └── student_data.csv   # Dataset (created by generate_data.py)
├── model/
│   ├── grade_prediction_model.pkl   # Trained model bundle (created by training)
│   └── metrics.json                 # Saved evaluation metrics
├── src/
│   ├── config.py          # Feature order, bands, ranges, paths (all config)
│   ├── preprocessing.py   # Load / clean / split (FR-08..FR-12)
│   ├── train_model.py     # Train + save model (FR-13,14,17)
│   ├── evaluate_model.py  # MAE / MSE / RMSE / R² (FR-16)
│   └── prediction.py      # Validate / predict / classify / explain (FR-18..24)
└── tests/
    └── test_core.py       # Test suite (SRS test cases TC-02..TC-06)
```

---

## Requirements

- Python 3.9+
- The packages listed in `requirements.txt` (pandas, numpy, scikit-learn,
  streamlit, matplotlib, joblib)

---

## Quick start

Run all commands from the project root (`Student-Grade-Predictor/`).

### 1. Create a virtual environment and install dependencies

Create and activate a virtual environment (this keeps the project's packages
isolated from your system Python), then install the requirements:

```bash
# Create the virtual environment (creates a "venv" folder)
python -m venv venv

# Activate it:
#   Windows (PowerShell):      venv\Scripts\Activate.ps1
#   Windows (Command Prompt):  venv\Scripts\activate.bat
#   macOS / Linux:             source venv/bin/activate

# Install dependencies into the active environment
pip install -r requirements.txt
```

You'll know it's active when your prompt is prefixed with `(venv)`. When you're
finished working, run `deactivate` to leave the environment.

> On Windows PowerShell, if activation is blocked by the execution policy, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` once in that
> terminal, then activate again.

### 2. Generate the dataset

The project ships without data (for privacy). Create a synthetic dataset of
~500 realistic records:

```bash
python generate_data.py
```

To use **your own data**, place a CSV at `data/student_data.csv` with these
columns (this exact order is expected):

```
study_hours,attendance,assignment_marks,previous_marks,internal_marks,final_score
```

### 3. Train the model

```bash
python -m src.train_model
```

This trains the Linear Regression model, prints the evaluation metrics and the
learned equation, and saves `model/grade_prediction_model.pkl` and
`model/metrics.json`.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Your browser opens the app (default: http://localhost:8501). Enter the student's
details and click **PREDICT SCORE**.

### 5. (Optional) Run the tests

```bash
pytest -q
```

---

## How it works

```
student_data.csv
      │  load + clean (preprocessing.py)
      ▼
   X (features) , y (final_score)
      │  train/test split (80/20)
      ▼
 LinearRegression().fit(X_train, y_train)     ← train_model.py
      │  evaluate on X_test (evaluate_model.py)
      ▼
 grade_prediction_model.pkl  +  metrics.json
      │  loaded by the app (prediction.py)
      ▼
 user input → validate → predict → classify → explain   ← app.py
```

### Input features (SRS §6.1)

| Feature | Range |
| --- | --- |
| Study Hours per Day | 0 – 15 |
| Attendance (%) | 0 – 100 |
| Assignment Marks | 0 – 100 |
| Previous Examination Marks | 0 – 100 |
| Internal Assessment Marks | 0 – 100 |

### Performance bands (SRS §3.6)

| Predicted score | Category |
| --- | --- |
| 90 – 100 | Excellent |
| 75 – 89 | Very Good |
| 60 – 74 | Good |
| 50 – 59 | Average |
| Below 50 | Needs Improvement |

These bands are configurable in `src/config.py` and are **not** an official
grading policy.

---

## Configuration

All tunable parameters live in `src/config.py`:

- `FEATURES` / `TARGET` — the ML schema and feature order
- `INPUT_RANGES` — validation bounds per feature
- `PERFORMANCE_BANDS` — score-to-category mapping
- `TEST_SIZE`, `RANDOM_STATE` — train/test split settings
- file paths for the dataset, model and metrics

---

## Ethics & limitations (SRS §29, §20)

- The predicted score is an **estimate**, not a guaranteed examination result.
- The model only uses five inputs; real performance also depends on motivation,
  wellbeing, teaching quality, resources and other factors it cannot see.
- Linear Regression assumes an approximately linear relationship between the
  features and the score; strongly non-linear patterns may not be captured.
- Prediction quality depends on the quality and representativeness of the data.
- The tool must **not** be the sole basis for academic decisions and should never
  be used to permanently label a student. A low prediction does not mean a
  student "will fail" — it is only the model's estimate given its training data.

---

## Possible future enhancements (SRS §21)

Comparing multiple algorithms (Decision Tree, Random Forest, Gradient Boosting),
pass/fail classification, at-risk student detection, a faculty dashboard,
database integration, authentication and historical performance tracking.

---

*Technology: Python · pandas · NumPy · scikit-learn · Streamlit · matplotlib*
