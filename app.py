"""
Student Grade Predictor - Streamlit dashboard.

Implements the web interface described in the SRS (Feature 8, section 7.1,
section 23) and satisfies FR-25 .. FR-30 and the usability / reliability
non-functional requirements (NFR-04 .. NFR-10).

Run from the project root:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.prediction import (
    classify_performance,
    explain_prediction,
    load_model,
    predict_score,
    validate_input,
)

# --------------------------------------------------------------------------- #
# Page configuration + light styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
)

# Colour per performance category (used for the result badge).
CATEGORY_COLORS = {
    "Excellent": "#1a7f37",
    "Very Good": "#2f81f7",
    "Good": "#0969da",
    "Average": "#bf8700",
    "Needs Improvement": "#cf222e",
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1150px; }
      .app-title { font-size: 2.1rem; font-weight: 800; margin-bottom: .2rem;
                   letter-spacing: -.5px; }
      .app-sub   { color: #57606a; font-size: 1.02rem; margin-bottom: .4rem; }
      .score-card { border: 1px solid #d0d7de; border-radius: 14px;
                    padding: 1.4rem 1.6rem; background: #f6f8fa; }
      .score-num { font-size: 3.2rem; font-weight: 800; line-height: 1; }
      .badge { display:inline-block; padding: .35rem .9rem; border-radius: 999px;
               color: #fff; font-weight: 700; font-size: 1.05rem; }
      .muted { color:#57606a; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once per session (NFR-02)."""
    return load_model()


@st.cache_data(show_spinner=False)
def get_dataset():
    """Load the dataset for the visualisations (returns None if absent)."""
    try:
        return pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="app-title">🎓 Student Grade Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Predict a student\'s expected final examination '
    "score from study habits and academic history, using a Linear Regression "
    "model.</div>",
    unsafe_allow_html=True,
)

# Load the model up front so a missing model shows a clear message (NFR-09).
try:
    bundle = get_model()
    model_ok = True
    model_error = ""
except FileNotFoundError as exc:
    bundle = None
    model_ok = False
    model_error = str(exc)

if not model_ok:
    st.error(
        "⚠️ The trained model could not be loaded, so predictions are "
        "unavailable.\n\n"
        f"**Details:** {model_error}\n\n"
        "**Fix:** open a terminal in the project folder and run:\n\n"
        "```\npython generate_data.py\npython -m src.train_model\n```"
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_predict, tab_data, tab_model, tab_about = st.tabs(
    ["🔮 Predict", "📊 Data Insights", "🧠 Model Info", "ℹ️ About"]
)


# ============================ PREDICT TAB ================================== #
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Enter Student Information")
        with st.form("prediction_form"):
            study_hours = st.number_input(
                config.FEATURE_LABELS["study_hours"],
                min_value=float(config.INPUT_RANGES["study_hours"][0]),
                max_value=float(config.INPUT_RANGES["study_hours"][1]),
                value=7.0, step=0.5,
                help="Average hours studied per day (0-15).",
            )
            attendance = st.number_input(
                config.FEATURE_LABELS["attendance"],
                min_value=float(config.INPUT_RANGES["attendance"][0]),
                max_value=float(config.INPUT_RANGES["attendance"][1]),
                value=90.0, step=1.0,
                help="Class attendance percentage (0-100).",
            )
            assignment_marks = st.number_input(
                config.FEATURE_LABELS["assignment_marks"],
                min_value=float(config.INPUT_RANGES["assignment_marks"][0]),
                max_value=float(config.INPUT_RANGES["assignment_marks"][1]),
                value=85.0, step=1.0,
                help="Average assignment marks (0-100).",
            )
            previous_marks = st.number_input(
                config.FEATURE_LABELS["previous_marks"],
                min_value=float(config.INPUT_RANGES["previous_marks"][0]),
                max_value=float(config.INPUT_RANGES["previous_marks"][1]),
                value=80.0, step=1.0,
                help="Previous examination marks (0-100).",
            )
            internal_marks = st.number_input(
                config.FEATURE_LABELS["internal_marks"],
                min_value=float(config.INPUT_RANGES["internal_marks"][0]),
                max_value=float(config.INPUT_RANGES["internal_marks"][1]),
                value=82.0, step=1.0,
                help="Internal assessment marks (0-100).",
            )
            submitted = st.form_submit_button("PREDICT SCORE",
                                              type="primary",
                                              use_container_width=True)

    with right:
        st.subheader("Prediction Result")
        if not submitted:
            st.info("Enter the student's details and click **PREDICT SCORE** "
                    "to see the estimated final score.")
        else:
            values = {
                "study_hours": study_hours,
                "attendance": attendance,
                "assignment_marks": assignment_marks,
                "previous_marks": previous_marks,
                "internal_marks": internal_marks,
            }
            is_valid, errors = validate_input(values)

            if not is_valid:
                # FR-24 / NFR-07: clear validation messages.
                st.error("Please correct the following before predicting:")
                for err in errors:
                    st.write(f"- {err}")
            else:
                score = predict_score(bundle, values)
                category = classify_performance(score)
                color = CATEGORY_COLORS.get(category, "#57606a")

                st.markdown(
                    f"""
                    <div class="score-card">
                      <div class="muted">Predicted Final Score</div>
                      <div class="score-num" style="color:{color}">
                        {score:.1f}<span style="font-size:1.3rem;color:#57606a">
                        &nbsp;/ 100</span></div>
                      <div style="margin-top:.8rem">
                        <span class="badge" style="background:{color}">
                        {category.upper()}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # --- Explainability (SRS 21.8) ------------------------------
                st.markdown("#### Why this prediction?")
                st.caption("How each factor pushes this student above (+) or "
                           "below (−) a typical student in the dataset.")
                contributions = explain_prediction(bundle, values)

                labels = [c["label"] for c in contributions][::-1]
                impacts = [c["impact"] for c in contributions][::-1]
                bar_colors = ["#1a7f37" if v >= 0 else "#cf222e" for v in impacts]

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(labels, impacts, color=bar_colors)
                ax.axvline(0, color="#57606a", linewidth=0.8)
                ax.set_xlabel("Impact on predicted score (points)")
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                positives = [c for c in contributions if c["impact"] > 0.05]
                negatives = [c for c in contributions if c["impact"] < -0.05]
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Major positive factors**")
                    if positives:
                        for c in positives:
                            st.write(f"✅ {c['label']} (+{c['impact']:.1f})")
                    else:
                        st.write("—")
                with cols[1]:
                    st.markdown("**Areas for improvement**")
                    if negatives:
                        for c in negatives:
                            st.write(f"⚠️ {c['label']} ({c['impact']:.1f})")
                    else:
                        st.write("—")

                st.caption("This score is an estimate produced by a model, not "
                           "a guaranteed result. See the About tab.")


# ============================ DATA INSIGHTS TAB ============================ #
with tab_data:
    st.subheader("Dataset Visualisations")
    df = get_dataset()
    if df is None:
        st.warning("Dataset not found. Run `python generate_data.py` to create it.")
    else:
        st.caption(f"Based on {len(df):,} student records in the training dataset.")

        feature_plots = [
            ("study_hours", "Study Hours vs Final Score"),
            ("attendance", "Attendance vs Final Score"),
            ("previous_marks", "Previous Marks vs Final Score"),
            ("assignment_marks", "Assignment Marks vs Final Score"),
        ]
        grid = st.columns(2)
        for i, (feat, title) in enumerate(feature_plots):
            with grid[i % 2]:
                fig, ax = plt.subplots(figsize=(5, 3.2))
                ax.scatter(df[feat], df[config.TARGET], s=12,
                           alpha=0.5, color="#2f81f7", edgecolors="none")
                ax.set_xlabel(config.FEATURE_LABELS[feat])
                ax.set_ylabel("Final Score")
                ax.set_title(title, fontsize=10)
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Actual vs predicted (uses the trained model on the whole dataset).
        st.markdown("#### Actual vs Predicted Scores")
        preds = bundle["model"].predict(df[config.FEATURES])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df[config.TARGET], preds, s=14, alpha=0.5,
                   color="#8250df", edgecolors="none")
        lo = min(df[config.TARGET].min(), preds.min())
        hi = max(df[config.TARGET].max(), preds.max())
        ax.plot([lo, hi], [lo, hi], color="#cf222e", linewidth=1.2,
                linestyle="--", label="Perfect prediction")
        ax.set_xlabel("Actual Final Score")
        ax.set_ylabel("Predicted Final Score")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


# ============================ MODEL INFO TAB ============================== #
with tab_model:
    st.subheader("Model Information")
    metrics = bundle["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['r2']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.2f}")
    c3.metric("MAE", f"{metrics['mae']:.2f}")
    c4.metric("MSE", f"{metrics['mse']:.2f}")

    st.write("")
    meta = pd.DataFrame(
        {
            "Property": ["Algorithm", "Target variable", "Training records",
                         "Testing records", "Trained at"],
            "Value": [
                bundle["algorithm"],
                bundle["target"],
                bundle["n_train"],
                bundle["n_test"],
                bundle["trained_at"],
            ],
        }
    )
    st.table(meta)

    st.markdown("#### Learned coefficients")
    st.caption("Each coefficient is the change in predicted final score per "
               "one-unit increase in that feature.")
    coefs = pd.DataFrame(
        {
            "Feature": list(bundle["coefficients"].keys()) + ["(intercept)"],
            "Coefficient": list(bundle["coefficients"].values())
            + [bundle["intercept"]],
        }
    )
    st.table(coefs)


# ============================ ABOUT TAB =================================== #
with tab_about:
    st.subheader("About this application")
    st.markdown(
        """
This tool estimates a student's expected **final examination score** from five
academic factors using an Ordinary Least Squares **Linear Regression** model
trained on historical (here, synthetic) student records.

**How to read the result**

The predicted score is mapped to a performance category using the bands below.
These are configurable project rules — **not** an official grading policy.
"""
    )
    bands = pd.DataFrame(
        {
            "Score range": ["90 – 100", "75 – 89", "60 – 74",
                            "50 – 59", "Below 50"],
            "Category": ["Excellent", "Very Good", "Good",
                         "Average", "Needs Improvement"],
        }
    )
    st.table(bands)

    st.markdown(
        """
**Important limitations & ethics (SRS §29)**

- Predictions are **estimates**, not guaranteed exam results.
- The model only sees the five inputs above; real performance also depends on
  motivation, wellbeing, teaching quality and many other factors.
- This should **not** be the sole basis for any academic decision, and should
  never be used to permanently label a student.
- A low predicted score (e.g. 48) does **not** mean a student "will fail" — it
  is only the model's estimate given the data it was trained on.
"""
    )
