"""Streamlit entry point: typography, warm design system, cached resources, and navigation router."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    PALETTE,
    SAVED_HMM_PATH,
    SAVED_SCALER_PATH,
)
from models.hmm_model import GaussianHMM
from preprocessing.loader import load_processed_or_synthetic
from preprocessing.split import SubjectScaler

# 1. Page Configuration
st.set_page_config(
    page_title="HAR | Forward & Backward HMM",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Warm Field-Notebook Typography and CSS
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..700&family=Source+Sans+3:ital,wght@0,300..700;1,300..700&display=swap');

/* Main Canvas Styling */
.stApp {{
    background-color: {PALETTE["background"]};
    color: {PALETTE["text_primary"]};
    font-family: 'Source Sans 3', sans-serif;
    font-size: 16px;
    line-height: 1.6;
}}

/* Typography: Fraunces Headings */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Fraunces', Georgia, serif !important;
    color: {PALETTE["text_primary"]} !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}}

/* Tabular Numbers */
.metric-value, .stMetricValue, td, th {{
    font-variant-numeric: tabular-nums;
}}

/* Paper Surface Cards */
.paper-card {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px rgba(46, 33, 24, 0.04);
}}

/* Callout Boxes */
.callout-box {{
    background-color: {PALETTE["surface"]};
    border-left: 4px solid {PALETTE["accent"]};
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}}

.callout-success {{
    border-left-color: {PALETTE["success"]};
}}

.callout-warning {{
    border-left-color: {PALETTE["warning"]};
}}

/* Divider */
hr {{
    border-top: 1px solid {PALETTE["border"]};
    margin: 1.5rem 0;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: {PALETTE["surface"]} !important;
    border-right: 1px solid {PALETTE["border"]};
}}

/* Buttons */
button[kind="primary"] {{
    background-color: {PALETTE["accent"]} !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}}

/* Metrics Cards */
div[data-testid="stMetric"] {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 4px rgba(46, 33, 24, 0.03);
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# 3. Cached Resource Loaders
@st.cache_resource(show_spinner=False)
def get_cached_model() -> GaussianHMM:
    """Load serialized trained HMM model."""
    if not SAVED_HMM_PATH.exists():
        from evaluation.train_and_evaluate import run_full_pipeline

        run_full_pipeline()
    return GaussianHMM.load(SAVED_HMM_PATH)


@st.cache_resource(show_spinner=False)
def get_cached_scaler() -> SubjectScaler:
    """Load serialized fitted SubjectScaler."""
    if not SAVED_SCALER_PATH.exists():
        from evaluation.train_and_evaluate import run_full_pipeline

        run_full_pipeline()
    return SubjectScaler.load(SAVED_SCALER_PATH)


@st.cache_data(show_spinner=False)
def get_cached_dataset() -> tuple[pd.DataFrame, bool]:
    """Load preprocessed PAMAP2 recording or synthetic fallback."""
    return load_processed_or_synthetic()


@st.cache_data(show_spinner=False)
def get_cached_evaluation_results() -> dict:
    """Load precomputed test and cross-validation evaluation results."""
    eval_path = Path(__file__).resolve().parent / "evaluation" / "evaluation_results.json"
    if not eval_path.exists():
        from evaluation.train_and_evaluate import run_full_pipeline

        return run_full_pipeline()

    with open(eval_path, encoding="utf-8") as f:
        return json.load(f)


# 4. Streamlit Navigation Router
pages = {
    "Overview": [
        st.Page("pages/01_Home.py", title="Home", icon="🏠"),
        st.Page("pages/02_Dataset_Explorer.py", title="Dataset Explorer", icon="📊"),
        st.Page("pages/03_Preprocessing.py", title="Preprocessing", icon="🔬"),
        st.Page("pages/04_Sensor_Visualisation.py", title="Sensor Dynamics", icon="📈"),
    ],
    "HMM Core & Algorithms": [
        st.Page("pages/05_HMM_Model.py", title="HMM Architecture", icon="📐"),
        st.Page("pages/06_Forward.py", title="Forward Algorithm", icon="➡️"),
        st.Page("pages/07_Backward.py", title="Backward Algorithm", icon="⬅️"),
        st.Page("pages/08_Posterior_Probabilities.py", title="Posterior Smoothing", icon="🎯"),
    ],
    "Inference & Benchmarks": [
        st.Page("pages/09_Activity_Recognition.py", title="Activity Recognition", icon="🏃"),
        st.Page("pages/10_Model_Evaluation.py", title="Model Evaluation", icon="🏆"),
        st.Page("pages/11_Beyond_the_Core.py", title="Beyond the Core", icon="✨"),
        st.Page("pages/12_About.py", title="About & Viva Voce", icon="🎓"),
    ],
}

pg = st.navigation(pages)
pg.run()
