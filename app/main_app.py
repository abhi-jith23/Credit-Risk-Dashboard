# app/main_app.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st
import yaml
from joblib import load

from app.ui_text import ABOUT_SIDEBAR_MD, APP_ICON, APP_TITLE
from app.pages.applicant_scoring import render as render_applicant
from app.pages.bi_metrics import render as render_bi_metrics
from app.pages.data_quality import render as render_data_quality
from app.pages.model_card import render as render_model_card
from app.pages.policy_impact import render as render_policy_impact
from app.pages.portfolio_explorer import render as render_portfolio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "config"
PIPELINE_YAML = CONFIG_DIR / "pipeline.yaml"
POLICY_YAML = CONFIG_DIR / "policy.yaml"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_project_path(rel: str | Path) -> Path:
    return PROJECT_ROOT / Path(rel)


@st.cache_data
def load_pipeline_cfg() -> dict[str, Any]:
    return _read_yaml(PIPELINE_YAML)


@st.cache_data
def list_model_names() -> list[str]:
    base = ARTIFACTS_DIR / "models"
    if not base.exists():
        return []
    names = sorted([p.name for p in base.iterdir() if p.is_dir()])
    return names


@st.cache_data
def list_model_versions(model_name: str) -> list[str]:
    base = ARTIFACTS_DIR / "models" / model_name
    if not base.exists():
        return []
    versions = sorted([p.name for p in base.iterdir() if p.is_dir()], reverse=True)
    return versions


@st.cache_resource
def load_model_pipeline(model_name: str, model_version: str):
    path = ARTIFACTS_DIR / "models" / model_name / model_version / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return load(path)


@st.cache_data
def load_model_metrics(model_name: str, model_version: str) -> dict:
    path = ARTIFACTS_DIR / "models" / model_name / model_version / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_model_feature_importance(model_name: str, model_version: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / "models" / model_name / model_version / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_model_test_predictions(model_name: str, model_version: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / "models" / model_name / model_version / "test_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_dataset_from_duckdb(duckdb_path_rel: str | Path) -> dict[str, Any]:
    """
    Loads dataset from DuckDB tables dim_applicant + fact_outcome.
    Returns a v1-compatible structure: X, y, df, numeric_cols, categorical_cols.
    """
    db_path = _resolve_project_path(duckdb_path_rel)
    if not db_path.exists():
        raise FileNotFoundError(f"Missing DuckDB database: {db_path}. Run: python -m src.warehouse")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT a.*, o.credit_risk
            FROM dim_applicant a
            JOIN fact_outcome o USING(application_id)
            ORDER BY application_id
            """
        ).df()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError("DuckDB returned 0 rows from dim_applicant/fact_outcome")

    # UI-friendly column names
    rename = {
        "status_checking_account": "checking_status",
        "savings_account_bonds": "savings_status",
        "present_employment_since": "employment_since",
        "installment_rate_pct_income": "installment_rate",
        "personal_status_and_sex": "personal_status_sex",
        "other_debtors_guarantors": "other_debtors",
        "present_residence_since": "residence_since",
        "age_years": "age",
        "existing_credits_count": "existing_credits",
        "people_liable_count": "num_dependents",
    }

    df_ui = df.rename(columns=rename).copy()
    df_ui["default"] = (df_ui["credit_risk"] == 2).astype(int)

    # y is 1 if default (bad), else 0
    y = df_ui["default"].astype(int)
    y.name = "default"

    X = df_ui.drop(columns=["default"]).copy()

    numeric_cols = [
        "duration_months",
        "credit_amount",
        "installment_rate",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    categorical_cols = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment_since",
        "personal_status_sex",
        "other_debtors",
        "property",
        "other_installment_plans",
        "housing",
        "job",
        "telephone",
        "foreign_worker",
    ]

    # Ensure categoricals are strings for UI controls
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)

    return {
        "X": X,
        "y": y,
        "df": df_ui,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def main() -> None:
    cfg = load_pipeline_cfg()

    # Resolve DuckDB path from config
    try:
        duckdb_path_rel = cfg["paths"]["duckdb_path"]
    except Exception as e:
        raise KeyError("config/pipeline.yaml missing required key: paths.duckdb_path") from e

    # -------- Sidebar: Model selection --------
    st.sidebar.title("Navigation")

    model_names = list_model_names()
    if not model_names:
        st.sidebar.error("No models found in artifacts/models/. Run: python -m src.train")
        st.stop()

    # Defaults from config/pipeline.yaml kpis section 
    default_model_name = str(cfg.get("kpis", {}).get("model_name", model_names[0]))
    if default_model_name not in model_names:
        default_model_name = model_names[0]

    model_name = st.sidebar.selectbox(
        "Model",
        options=model_names,
        index=model_names.index(default_model_name),
    )

    versions = list_model_versions(model_name)
    if not versions:
        st.sidebar.error(f"No versions found for model '{model_name}'. Run: python -m src.train")
        st.stop()

    default_version = str(cfg.get("kpis", {}).get("model_version", versions[0]))
    if default_version not in versions:
        default_version = versions[0]

    model_version = st.sidebar.selectbox(
        "Model version",
        options=versions,
        index=versions.index(default_version),
    )

    # -------- Sidebar: Page selection --------
    page = st.sidebar.radio(
        "Go to",
        [
            "Applicant Scoring",
            "Portfolio Explorer",
            "Model Card",
            "Data Quality",
            "BI Metrics",
            "Policy Impact",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(ABOUT_SIDEBAR_MD)

    # -------- Load resources --------
    try:
        pipeline = load_model_pipeline(model_name, model_version)
    except Exception as e:
        st.error(f"Could not load model pipeline. Details: {e}")
        st.stop()

    data = load_dataset_from_duckdb(duckdb_path_rel)
    metrics = load_model_metrics(model_name, model_version)
    feat_imp = load_model_feature_importance(model_name, model_version)
    test_preds = load_model_test_predictions(model_name, model_version)

    # -------- Header --------
    st.title(APP_TITLE)
    st.caption("Loan Underwriting Tool • Predict Probability of Default (PD) from applicant features")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    c1.markdown("**Selected model**")
    c1.write(f"{model_name} • {model_version}")
    c2.metric("DuckDB", "Loaded" if (_resolve_project_path(duckdb_path_rel)).exists() else "Missing")
    if metrics:
        c3.metric("ROC-AUC (test)", f"{metrics.get('roc_auc', float('nan')):.3f}")
        c4.metric("Default rate (test)", f"{metrics.get('default_rate', float('nan')):.2%}")
    else:
        c3.metric("ROC-AUC (test)", "—")
        c4.metric("Default rate (test)", "—")

    st.markdown("---")

    ctx = {
        "project_root": PROJECT_ROOT,
        "artifacts_dir": ARTIFACTS_DIR,
        "config_dir": CONFIG_DIR,
        "pipeline_cfg": cfg,
        "policy_yaml_path": POLICY_YAML,
        "duckdb_path": _resolve_project_path(duckdb_path_rel),
        "sql_ddl_dir": _resolve_project_path(cfg["paths"]["sql_ddl_dir"]),
        "sql_kpis_dir": _resolve_project_path(cfg["paths"]["sql_kpis_dir"]),
        "model": {"name": model_name, "version": model_version},
        "model_options": model_names,
        "model_versions": {mn: list_model_versions(mn) for mn in model_names},
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
    elif page == "Model Card":
        render_model_card(ctx)
    elif page == "Data Quality":
        render_data_quality(ctx)
    elif page == "BI Metrics":
        render_bi_metrics(ctx)
    else:
        render_policy_impact(ctx)


if __name__ == "__main__":
    main()
