# app/main_app.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from src.data import load_german_credit
from app.ui_text import APP_TITLE, APP_ICON, ABOUT_SIDEBAR_MD
from app.pages.applicant_scoring import render as render_applicant
from app.pages.portfolio_explorer import render as render_portfolio
from app.pages.model_card import render as render_model_card


# Streamlit recommends calling set_page_config first in the script. :contentReference[oaicite:1]{index=1}
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"


@st.cache_resource
def load_pipeline():
    """Cache the model pipeline so it loads once per app session/process. :contentReference[oaicite:2]{index=2}"""
    path = ARTIFACTS_DIR / "credit_risk_pipeline.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return load(path)


@st.cache_data
def load_metrics_dict() -> dict:
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_feature_importance_df() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_test_predictions_df() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "test_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_dataset() -> dict:
    bundle = load_german_credit(data_dir=DATA_DIR)
    df_full = bundle.X.copy()
    df_full["default"] = bundle.y.values
    return {
        "X": bundle.X,
        "y": bundle.y,
        "df": df_full,
        "numeric_cols": bundle.numeric_cols,
        "categorical_cols": bundle.categorical_cols,
    }


def main():
    # ---- Load resources (cached) ----
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Could not load model pipeline. Details: {e}")
        st.stop()

    data = load_dataset()
    metrics = load_metrics_dict()
    feat_imp = load_feature_importance_df()
    test_preds = load_test_predictions_df()

    # ---- Sidebar nav (friend-style: single app + sidebar pages) ----
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Applicant Scoring", "Portfolio Explorer", "Model Card"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(ABOUT_SIDEBAR_MD)

    # ---- Header ----
    st.title(APP_TITLE)
    st.caption("Loan Underwriting Tool • Predict Probability of Default (PD) from applicant features")

    # Quick “status” strip on the home top
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    c1.markdown("**Model**")
    c1.write("Logistic Regression (Pipeline)")
    c2.metric("Artifacts", "Loaded" if metrics else "Partially loaded")
    if metrics:
        c3.metric("ROC-AUC (test)", f"{metrics.get('roc_auc', float('nan')):.3f}")
        c4.metric("Default rate (test)", f"{metrics.get('default_rate', float('nan')):.2%}")
    else:
        c3.metric("ROC-AUC (test)", "—")
        c4.metric("Default rate (test)", "—")

    st.markdown("---")

    # ---- Route to pages ----
    ctx = {
        "project_root": PROJECT_ROOT,
        "artifacts_dir": ARTIFACTS_DIR,
        "data_dir": DATA_DIR,
        "pipeline": pipeline,
        "data": data,
        "metrics": metrics,
        "feature_importance": feat_imp,
        "test_predictions": test_preds,
    }

    if page == "Applicant Scoring":
        render_applicant(ctx)
    elif page == "Portfolio Explorer":
        render_portfolio(ctx)
    else:
        render_model_card(ctx)


if __name__ == "__main__":
    main()
