import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Learner Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
BANNER_PATH = BASE_DIR / "banner.png"

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: #f5f8ff; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .hero {
        background: linear-gradient(115deg, #102a67 0%, #253f9b 58%, #6b42c1 100%);
        padding: 1.6rem 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(28, 52, 120, .14);
    }
    .hero h1 { color: white; font-size: 2.1rem; margin: 0 0 .35rem 0; }
    .hero p { color: #e5ebff; font-size: 1rem; margin: 0; }
    .section-title {
        color: #172554; font-size: 1.12rem; font-weight: 700;
        padding: .35rem 0; border-bottom: 1px solid #dbe5ff;
        margin: .5rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e3eafe;
        padding: 14px 16px; border-radius: 14px;
        box-shadow: 0 3px 12px rgba(30, 64, 175, .05);
    }
    .result-card {
        background: white; border: 1px solid #e3eafe;
        padding: 1.25rem; border-radius: 16px;
        box-shadow: 0 4px 16px rgba(30, 64, 175, .06);
    }
    .small-note { color: #64748b; font-size: .86rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

if not MODEL_PATH.exists():
    st.error("model.pkl was not found. Place model.pkl in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load model.pkl: {error}")
    st.info("Use the same Python package versions used when you trained and saved the model.")
    st.stop()

# Read categories from the fitted preprocessing pipeline.
# This avoids guessing category names and keeps the app aligned with training.
try:
    preprocessing = model.named_steps["preprocessing"]
    nominal_pipeline = preprocessing.named_transformers_["Nominal Transformer"]
    ordinal_pipeline = preprocessing.named_transformers_["Ordinal Transformer"]

    device_options = list(nominal_pipeline.named_steps["Encoding"].categories_[1])
    course_options = list(nominal_pipeline.named_steps["Encoding"].categories_[0])
    level_options = list(ordinal_pipeline.named_steps["Encoding"].categories_[0])
except Exception:
    st.error(
        "The model's fitted preprocessing steps could not be read. "
        "Check that model.pkl is the complete fitted Pipeline saved from your notebook."
    )
    st.stop()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), use_container_width=True)

st.markdown("""
<div class="hero">
    <h1>🎓 Online Course Learner Dropout Prediction</h1>
    <p>Identify learners who may be at risk of dropping out using course activity,
    engagement and performance information.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Enter a learner's course and activity details below, then select "
    "**Predict dropout risk** to get the model's estimate."
)

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
with st.form("learner_prediction_form"):
    st.markdown('<div class="section-title">📚 Course Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        course_category = st.selectbox("Course category", course_options)
    with c2:
        course_level = st.selectbox("Course level", level_options)
    with c3:
        primary_device = st.selectbox("Primary device", device_options)

    st.markdown('<div class="section-title">📊 Learning Engagement</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        course_price_inr = st.number_input(
            "Course price (₹)", min_value=0.0, max_value=21668.0,
            value=0.0, step=100.0
        )
        course_duration_weeks = st.number_input(
            "Course duration (weeks)", min_value=4, max_value=16,
            value=10, step=1
        )
        avg_weekly_logins = st.number_input(
            "Average weekly logins", min_value=0.0, max_value=57.3,
            value=4.4, step=0.1
        )

    with c2:
        total_watch_minutes = st.number_input(
            "Total watch time (minutes)", min_value=3.0, max_value=9786.0,
            value=226.7, step=10.0
        )
        video_completion_pct = st.slider(
            "Video completion (%)", min_value=0.0, max_value=100.0,
            value=55.0, step=0.5
        )
        avg_quiz_score = st.slider(
            "Average quiz score (%)", min_value=0.0, max_value=100.0,
            value=62.0, step=0.5
        )

    with c3:
        assignment_submission_pct = st.slider(
            "Assignment submission (%)", min_value=0.0, max_value=100.0,
            value=44.5, step=0.5
        )
        avg_submission_delay_days = st.number_input(
            "Average submission delay (days)", min_value=0.0, max_value=33.3,
            value=3.0, step=0.1
        )

    submitted = st.form_submit_button(
        "🔍 Predict dropout risk", use_container_width=True, type="primary"
    )

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "primary_device": primary_device,
        "course_category": course_category,
        "course_level": course_level,
        "course_price_inr": course_price_inr,
        "course_duration_weeks": course_duration_weeks,
        "avg_weekly_logins": avg_weekly_logins,
        "total_watch_minutes": total_watch_minutes,
        "video_completion_pct": video_completion_pct,
        "avg_quiz_score": avg_quiz_score,
        "assignment_submission_pct": assignment_submission_pct,
        "avg_submission_delay_days": avg_submission_delay_days
    }])

    try:
        prediction = int(model.predict(input_data)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            dropout_index = classes.index(1)
            dropout_probability = float(probabilities[dropout_index])
        else:
            st.error("This model does not provide predict_proba(), so a risk probability cannot be displayed.")
            st.stop()

        st.markdown("---")
        st.markdown('<div class="section-title">📌 Prediction Result</div>', unsafe_allow_html=True)

        result_col, metric_col = st.columns([1.15, 1])

        with result_col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            if prediction == 1:
                st.error("⚠️ Model prediction: At risk of dropping out")
                st.markdown(
                    "The model classified this learner as **likely to drop out**. "
                    "Consider offering timely support and checking their learning activity."
                )
            else:
                st.success("✅ Model prediction: Not classified as a dropout")
                st.markdown(
                    "The model classified this learner as **not likely to drop out**. "
                    "Continue monitoring engagement and course progress."
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with metric_col:
            st.metric("Estimated dropout probability", f"{dropout_probability:.1%}")
            st.progress(max(0.0, min(1.0, dropout_probability)))
            st.caption("Probability estimated by the trained Logistic Regression classification pipeline.")

        st.markdown("#### Learner details submitted")
        st.dataframe(input_data, use_container_width=True, hide_index=True)

        st.markdown(
            '<p class="small-note">This is a machine-learning estimate, not a certainty. '
            'Use it as decision support alongside human review.</p>',
            unsafe_allow_html=True
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that all input categories match the categories used during training "
            "and that the saved model is compatible with your installed libraries."
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:.85rem;'>"
    "Built with Python • Scikit-learn • Logistic Regression• Streamlit"
    "<br>Online Course Learner Dropout Prediction</div>",
    unsafe_allow_html=True
)


