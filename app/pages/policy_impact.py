# app/pages/policy_impact.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


def _ensure_policy_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_run_summary (
            policy_run_id VARCHAR PRIMARY KEY,
            run_at_utc VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            policy_version VARCHAR,
            t_low DOUBLE,
            t_high DOUBLE,
            review_capacity_per_day INTEGER,
            review_approval_rate DOUBLE,
            profit_good DOUBLE,
            loss_default DOUBLE,
            review_cost DOUBLE,
            n_total INTEGER,
            n_approve INTEGER,
            n_review INTEGER,
            n_review_processed INTEGER,
            n_review_overflow INTEGER,
            n_decline INTEGER,
            default_rate_approve DOUBLE,
            total_expected_value DOUBLE,
            expected_value_per_app DOUBLE
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_run_decisions (
            policy_run_id VARCHAR,
            application_id INTEGER,
            pd DOUBLE,
            true_default INTEGER,
            decision_bucket VARCHAR,
            review_processed INTEGER,
            expected_value DOUBLE
        );
        """
    )


def render(ctx: dict) -> None:
    st.subheader("Policy Impact Simulator")

    db_path: Path = ctx["duckdb_path"]
    if not db_path.exists():
        st.error(f"Missing DuckDB: {db_path}. Run: python -m src.warehouse")
        st.stop()

    model_options: list[str] = ctx["model_options"]
    versions_map: dict[str, list[str]] = ctx["model_versions"]

    default_model = ctx["model"]["name"]
    model_name = st.selectbox("Model", options=model_options, index=model_options.index(default_model))

    versions = versions_map.get(model_name, [])
    if not versions:
        st.error(f"No versions found for model '{model_name}'. Run: python -m src.train")
        st.stop()

    default_version = ctx["model"]["version"] if ctx["model"]["name"] == model_name else versions[0]
    if default_version not in versions:
        default_version = versions[0]

    model_version = st.selectbox("Model version", options=versions, index=versions.index(default_version))

    st.markdown("### Policy parameters")
    policy_version = st.text_input("Policy version label", value="policy_v1")

    t_low, t_high = st.slider(
        "Thresholds (t_low, t_high)",
        min_value=0.0,
        max_value=1.0,
        value=(0.20, 0.35),
        step=0.01,
    )

    review_capacity = st.number_input("Review capacity per day", min_value=1, value=50, step=1)
    review_approval_rate = st.slider("Review approval rate", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    profit_good = st.number_input("Profit if good (approved)", value=1000.0, step=50.0)
    loss_default = st.number_input("Loss if default (approved)", value=5000.0, step=50.0)
    review_cost = st.number_input("Cost per processed review", value=20.0, step=1.0)

    if not (0.0 <= t_low < t_high <= 1.0):
        st.error("Threshold rule must satisfy: 0.0 <= t_low < t_high <= 1.0")
        st.stop()

    if st.button("Run policy simulation (write to DuckDB)"):
        con = duckdb.connect(str(db_path))
        try:
            _ensure_policy_tables(con)

            df = con.execute(
                """
                SELECT s.application_id, s.pd, o.credit_risk
                FROM fact_score s
                JOIN fact_outcome o USING(application_id)
                WHERE s.model_name = ? AND s.model_version = ?
                ORDER BY s.application_id
                """,
                [model_name, model_version],
            ).df()

            if df.empty:
                st.error(
                    "No scores found for this model_name/model_version in fact_score.\n"
                    "Run: python -m src.train"
                )
                st.stop()

            df["true_default"] = (df["credit_risk"] == 2).astype(int)

            def bucket(pd_val: float) -> str:
                if pd_val < t_low:
                    return "APPROVE"
                if pd_val < t_high:
                    return "REVIEW"
                return "DECLINE"

            df["decision_bucket"] = df["pd"].astype(float).apply(bucket)

            # Deterministic review capacity handling:
            review_mask = df["decision_bucket"] == "REVIEW"
            review_df = df.loc[review_mask].sort_values("pd", ascending=True).copy()

            review_df["review_processed"] = 0
            review_df.iloc[: int(review_capacity), review_df.columns.get_loc("review_processed")] = 1

            df["review_processed"] = 0
            df.loc[review_df.index, "review_processed"] = review_df["review_processed"]

            df.loc[(df["decision_bucket"] == "REVIEW") & (df["review_processed"] == 0), "decision_bucket"] = "REVIEW_OVERFLOW"

            approve_value = (1 - df["true_default"]) * profit_good - df["true_default"] * loss_default

            df["expected_value"] = 0.0
            df.loc[df["decision_bucket"] == "APPROVE", "expected_value"] = approve_value[df["decision_bucket"] == "APPROVE"]

            processed_review_mask = (df["decision_bucket"] == "REVIEW") & (df["review_processed"] == 1)
            df.loc[processed_review_mask, "expected_value"] = (
                review_approval_rate * approve_value[processed_review_mask] - review_cost
            )

            n_total = int(len(df))
            n_approve = int((df["decision_bucket"] == "APPROVE").sum())
            n_review = int((df["decision_bucket"] == "REVIEW").sum())
            n_review_processed = int(((df["decision_bucket"] == "REVIEW") & (df["review_processed"] == 1)).sum())
            n_review_overflow = int((df["decision_bucket"] == "REVIEW_OVERFLOW").sum())
            n_decline = int((df["decision_bucket"] == "DECLINE").sum())

            default_rate_approve = float(df.loc[df["decision_bucket"] == "APPROVE", "true_default"].mean()) if n_approve > 0 else 0.0
            total_expected_value = float(df["expected_value"].sum())
            expected_value_per_app = float(total_expected_value / n_total)

            run_at = datetime.now(timezone.utc).isoformat()
            policy_run_id = f"{run_at.replace(':','').replace('-','')}_{policy_version}_{model_name}_{model_version}"

            con.execute("DELETE FROM policy_run_summary WHERE policy_run_id = ?;", [policy_run_id])
            con.execute(
                """
                INSERT INTO policy_run_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    policy_run_id,
                    run_at,
                    model_name,
                    model_version,
                    policy_version,
                    float(t_low),
                    float(t_high),
                    int(review_capacity),
                    float(review_approval_rate),
                    float(profit_good),
                    float(loss_default),
                    float(review_cost),
                    n_total,
                    n_approve,
                    n_review,
                    n_review_processed,
                    n_review_overflow,
                    n_decline,
                    default_rate_approve,
                    total_expected_value,
                    expected_value_per_app,
                ],
            )

            out_df = df[["application_id", "pd", "true_default", "decision_bucket", "review_processed", "expected_value"]].copy()
            out_df.insert(0, "policy_run_id", policy_run_id)

            con.register("out_df", out_df)
            con.execute("DELETE FROM policy_run_decisions WHERE policy_run_id = ?;", [policy_run_id])
            con.execute(
                """
                INSERT INTO policy_run_decisions
                SELECT policy_run_id, application_id, pd, true_default, decision_bucket, review_processed, expected_value
                FROM out_df
                """
            )

            st.success("Policy run written to DuckDB.")
            st.code(policy_run_id, language="text")

            st.markdown("### Summary")
            st.metric("Total expected value", f"{total_expected_value:,.2f}")
            st.metric("Expected value per application", f"{expected_value_per_app:,.2f}")

            st.markdown("### Counts")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Approve", n_approve)
            c2.metric("Review", n_review)
            c3.metric("Review overflow", n_review_overflow)
            c4.metric("Decline", n_decline)

            st.markdown("### Sample decisions")
            st.dataframe(out_df.head(30), use_container_width=True)

        finally:
            con.close()
