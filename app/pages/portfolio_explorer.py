# app/pages/portfolio_explorer.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_text import FEATURE_LABELS, format_category_value  # add


def render(ctx: dict):
    st.subheader("Portfolio Explorer (Dataset + Risk Segmentation)")

    pipeline = ctx["pipeline"]
    df_full: pd.DataFrame = ctx["data"]["df"].copy()
    categorical_cols: list[str] = ctx["data"]["categorical_cols"]

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

    X = df.drop(columns=["default"])
    df["pd_default"] = pipeline.predict_proba(X)[:, 1]

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
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No rows after filters.")

    with right:
        st.markdown("#### Credit amount distribution")
        if len(df) and "credit_amount" in df.columns:
            fig = px.histogram(df, x="credit_amount", nbins=30)
            fig.update_layout(xaxis_title="Credit Amount", yaxis_title="Count")
            st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("#### Data preview")
    st.dataframe(df.head(50), width="stretch")
