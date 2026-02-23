# app/pages/applicant_scoring.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui_text import FEATURE_HELP, FEATURE_LABELS, format_category_value


# UI column -> model-training column (warehouse)
UI_TO_MODEL = {
    "checking_status": "status_checking_account",
    "duration_months": "duration_months",
    "credit_history": "credit_history",
    "purpose": "purpose",
    "credit_amount": "credit_amount",
    "savings_status": "savings_account_bonds",
    "employment_since": "present_employment_since",
    "installment_rate": "installment_rate_pct_income",
    "personal_status_sex": "personal_status_and_sex",
    "other_debtors": "other_debtors_guarantors",
    "residence_since": "present_residence_since",
    "property": "property",
    "age": "age_years",
    "other_installment_plans": "other_installment_plans",
    "housing": "housing",
    "existing_credits": "existing_credits_count",
    "job": "job",
    "num_dependents": "people_liable_count",
    "telephone": "telephone",
    "foreign_worker": "foreign_worker",
}


def _decision(pd_default: float, band: tuple[float, float]) -> str:
    low, high = band
    if pd_default < low:
        return "✅ Approve"
    if pd_default < high:
        return "🟨 Manual Review"
    return "⛔ Decline"


def _to_model_frame(user_ui: dict) -> pd.DataFrame:
    user_model = {UI_TO_MODEL.get(k, k): v for k, v in user_ui.items()}
    return pd.DataFrame([user_model])


def render(ctx: dict):
    st.subheader("Applicant Scoring (Underwriting)")

    pipeline = ctx["pipeline"]
    model_name = ctx["model"]["name"]
    model_version = ctx["model"]["version"]

    df_full: pd.DataFrame = ctx["data"]["df"]
    numeric_cols: list[str] = ctx["data"]["numeric_cols"]
    categorical_cols: list[str] = ctx["data"]["categorical_cols"]

    st.caption(f"Scoring using: {model_name} • {model_version}")

    # ---- Session state ----
    if "has_scored" not in st.session_state:
        st.session_state["has_scored"] = False
        st.session_state["last_user"] = None
        st.session_state["last_pd_default"] = None

    # Sidebar decision thresholds (UI only)
    st.sidebar.markdown("### Decision thresholds (UI)")
    band = st.sidebar.slider(
        "Manual review band (PD range)",
        min_value=0.0,
        max_value=1.0,
        value=(0.20, 0.35),
        step=0.01,
        help="PD < lower → Approve, between → Manual Review, above upper → Decline",
    )

    st.markdown("Enter applicant details and click **Score applicant**.")

    # Defaults from dataset modes / medians
    defaults_cat = {c: str(df_full[c].mode(dropna=True).iloc[0]) for c in categorical_cols}
    defaults_num = {c: float(df_full[c].median()) for c in numeric_cols}

    with st.form("score_form", clear_on_submit=False):
        st.markdown("#### Inputs")
        colA, colB = st.columns(2)

        user = {}

        with colA:
            st.markdown("**Numeric**")
            for c in numeric_cols:
                vmin = float(df_full[c].min())
                vmax = float(df_full[c].max())
                step = 1.0 if c != "credit_amount" else 50.0
                user[c] = st.number_input(
                    label=FEATURE_LABELS.get(c, c),
                    min_value=vmin,
                    max_value=vmax,
                    value=float(defaults_num[c]),
                    step=step,
                    help=FEATURE_HELP.get(c, ""),
                )

        with colB:
            st.markdown("**Categorical**")
            for c in categorical_cols:
                opts = sorted(df_full[c].astype(str).unique().tolist())
                user[c] = st.selectbox(
                    label=FEATURE_LABELS.get(c, c),
                    options=opts,
                    index=opts.index(defaults_cat[c]) if defaults_cat[c] in opts else 0,
                    format_func=lambda x, c=c: format_category_value(c, x),
                )

        submitted = st.form_submit_button("Score applicant")

    if submitted:
        X_user_now = _to_model_frame(user)
        pd_now = float(pipeline.predict_proba(X_user_now)[0, 1])
        st.session_state["has_scored"] = True
        st.session_state["last_user"] = user
        st.session_state["last_pd_default"] = pd_now

    if not st.session_state["has_scored"] or st.session_state["last_user"] is None:
        st.info("Fill the inputs and click **Score applicant** to compute PD.")
        return

    base_user = dict(st.session_state["last_user"])
    pd_default = float(st.session_state["last_pd_default"])
    decision = _decision(pd_default, band)

    r1, r2, r3 = st.columns(3)
    r1.metric("PD (probability of default)", f"{pd_default:.3f}")
    r2.metric("Decision", decision)
    r3.metric("Risk band", f"[{band[0]:.2f}, {band[1]:.2f}]")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_default,
            number={"valueformat": ".3f"},
            gauge={"axis": {"range": [0, 1]}},
            title={"text": "PD Gauge"},
        )
    )
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown("---")
    st.markdown("### What-if simulation")

    what_col = st.selectbox(
        "Vary one input to see how PD changes",
        options=["credit_amount", "duration_months", "age"],
        format_func=lambda x: FEATURE_LABELS.get(x, x),
        key="what_if_feature",
    )

    vmin = float(df_full[what_col].min())
    vmax = float(df_full[what_col].max())
    n_points = 25

    grid = np.linspace(vmin, vmax, n_points)
    if what_col != "credit_amount":
        grid = np.round(grid).astype(int)
    else:
        grid = (np.round(grid / 50) * 50).astype(int)

    rows_model = []
    for g in grid:
        row_ui = base_user.copy()
        row_ui[what_col] = float(g)
        rows_model.append({UI_TO_MODEL.get(k, k): v for k, v in row_ui.items()})

    X_grid = pd.DataFrame(rows_model)
    pd_grid = pipeline.predict_proba(X_grid)[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=pd_grid, mode="lines+markers", name="PD"))
    fig.update_layout(
        xaxis_title=FEATURE_LABELS.get(what_col, what_col),
        yaxis_title="Predicted PD",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)
