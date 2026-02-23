# app/pages/portfolio_explorer.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_text import FEATURE_LABELS, format_category_value


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


def _to_model_frame(df_ui: pd.DataFrame, categorical_cols_ui: list[str]) -> pd.DataFrame:
    """
    Convert the UI dataframe (friendly names) into the model dataframe (warehouse names),
    with correct dtypes for the sklearn pipeline.
    """
    feature_cols_ui = [c for c in UI_TO_MODEL.keys() if c in df_ui.columns]
    X_ui = df_ui[feature_cols_ui].copy()
    X_model = X_ui.rename(columns=UI_TO_MODEL)

    # Cast categoricals to string (important for OneHotEncoder in trained pipeline)
    model_cat_cols = [UI_TO_MODEL[c] for c in categorical_cols_ui if c in UI_TO_MODEL]
    for c in model_cat_cols:
        if c in X_model.columns:
            X_model[c] = X_model[c].astype(str)

    return X_model


def render(ctx: dict):
    st.subheader("Portfolio Explorer (Dataset + Risk Segmentation)")

    pipeline = ctx["pipeline"]
    model_name = ctx["model"]["name"]
    model_version = ctx["model"]["version"]

    df_full: pd.DataFrame = ctx["data"]["df"].copy()
    categorical_cols: list[str] = ctx["data"]["categorical_cols"]

    st.caption(f"Scoring using: {model_name} • {model_version}")

    st.sidebar.markdown("### Filters")

    filter_cols = ["purpose", "housing", "employment_since", "checking_status"]
    filter_cols = [c for c in filter_cols if c in categorical_cols]

    filters = {}
    for c in filter_cols:
        opts = sorted(df_full[c].astype(str).unique().tolist())
        sel = st.sidebar.multiselect(
            label=FEATURE_LABELS.get(c, c),
            options=opts,
            default=[],
            format_func=lambda x, c=c: format_category_value(c, x),
        )
        filters[c] = sel

    df = df_full.copy()
    for c, sel in filters.items():
        if sel:
            df = df[df[c].astype(str).isin(sel)]

    if len(df):
        X_model = _to_model_frame(df, categorical_cols_ui=categorical_cols)
        df["pd_default"] = pipeline.predict_proba(X_model)[:, 1]
    else:
        df["pd_default"] = pd.Series(dtype=float)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Observed default rate", f"{df['default'].mean():.2%}" if len(df) else "—")
    c3.metric("Average predicted PD", f"{df['pd_default'].mean():.3f}" if len(df) else "—")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("#### PD distribution")
        if len(df):
            fig = px.histogram(df, x="pd_default", nbins=30)
            fig.update_layout(xaxis_title="Predicted PD", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No rows after filters.")

    with right:
        st.markdown("#### Credit amount distribution")
        if len(df) and "credit_amount" in df.columns:
            fig = px.histogram(df, x="credit_amount", nbins=30)
            fig.update_layout(xaxis_title="Credit Amount", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Default rate by category")
    cat = st.selectbox(
        "Choose a categorical feature",
        options=filter_cols or categorical_cols,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
    )

    if len(df):
        grp = (
            df.groupby(cat, dropna=False)["default"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"default": "default_rate"})
        )
        fig = px.bar(grp, x=cat, y="default_rate")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Data preview")
    st.dataframe(df.head(50), use_container_width=True)
