# app/pages/model_card.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def render(ctx: dict):
    model_name = ctx["model"]["name"]
    model_version = ctx["model"]["version"]

    st.subheader("Model Card (Governance-style view)")
    st.caption(f"Model: {model_name} • {model_version}")

    metrics: dict = ctx.get("metrics", {}) or {}
    feat_imp: pd.DataFrame = ctx.get("feature_importance", pd.DataFrame())
    test_preds: pd.DataFrame = ctx.get("test_predictions", pd.DataFrame())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC (test)", f"{metrics.get('roc_auc', float('nan')):.3f}" if metrics else "—")
    k2.metric("Avg Precision (test)", f"{metrics.get('avg_precision', float('nan')):.3f}" if metrics else "—")
    k3.metric("Brier (test)", f"{metrics.get('brier', float('nan')):.3f}" if metrics else "—")
    k4.metric("Default rate (test)", f"{metrics.get('default_rate', float('nan')):.2%}" if metrics else "—")

    st.markdown("---")

    st.markdown("### ROC Curve")
    roc = metrics.get("roc_curve", {})
    if roc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random"))
        fig.update_layout(
            xaxis_title="FPR",
            yaxis_title="TPR",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ROC curve not found in metrics.json.")

    st.markdown("### Confusion Matrix (choose threshold)")
    if not test_preds.empty:
        thr = st.slider("Classification threshold", 0.0, 1.0, 0.50, 0.01)
        y_true = test_preds["y_true"].astype(int).values
        y_proba = test_preds["pd_default"].astype(float).values
        y_hat = (y_proba >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)

        a1, a2, a3 = st.columns(3)
        a1.metric("Accuracy", f"{acc:.3f}")
        a2.metric("Precision", f"{prec:.3f}")
        a3.metric("Recall", f"{rec:.3f}")

        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("test_predictions.csv not found or empty — cannot compute confusion matrix interactively.")

    st.markdown("---")

    st.markdown("### Calibration")
    if not test_preds.empty:
        y_true = test_preds["y_true"].astype(int).values
        y_proba = test_preds["pd_default"].astype(float).values
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect"))
        fig.update_layout(xaxis_title="Mean predicted PD", yaxis_title="Observed default rate")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Precision–Recall Curve")
    if not test_preds.empty:
        y_true = test_preds["y_true"].astype(int).values
        y_proba = test_preds["pd_default"].astype(float).values
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### Global Feature Importance")

    if feat_imp is None or feat_imp.empty:
        st.info("feature_importance.csv not found or empty.")
        return

    # LightGBM file uses "importance"
    if {"feature", "importance"}.issubset(feat_imp.columns):
        topn = st.slider("How many features to show", 10, 50, 20, 5)
        top = feat_imp.sort_values("importance", ascending=False).head(topn)
        fig = px.bar(top[::-1], x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top, use_container_width=True)
        return

    # LogReg file uses "coef" + "abs_coef"
    if {"feature", "coef", "abs_coef"}.issubset(feat_imp.columns):
        topn = st.slider("How many features to show", 10, 50, 20, 5)
        top = feat_imp.sort_values("abs_coef", ascending=False).head(topn)
        fig = px.bar(top[::-1], x="coef", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top, use_container_width=True)
        return

    st.info("feature_importance.csv columns do not match expected formats for logreg or lightgbm.")
