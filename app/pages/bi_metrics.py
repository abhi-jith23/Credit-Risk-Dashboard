# app/pages/bi_metrics.py
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


def _df(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    if params is None:
        params = []
    return con.execute(sql, params).df()


def _render_sql(sql: str, params: dict[str, str]) -> str:
    for k, v in params.items():
        sql = sql.replace(f"{{{{{k}}}}}", v)
    return sql


def render(ctx: dict) -> None:
    st.subheader("BI Metrics (DuckDB KPI tables)")

    db_path: Path = ctx["duckdb_path"]
    sql_ddl_dir: Path = ctx["sql_ddl_dir"]
    sql_kpis_dir: Path = ctx["sql_kpis_dir"]

    model_name = ctx["model"]["name"]
    model_version = ctx["model"]["version"]

    if not db_path.exists():
        st.error(f"Missing DuckDB: {db_path}. Run: python -m src.warehouse")
        st.stop()

    con = duckdb.connect(str(db_path))
    try:
        # Let user pick a policy_run_id from policy_run_summary (most recent first)
        summary = _df(
            con,
            """
            SELECT policy_run_id, run_at_utc, model_name, model_version, policy_version,
                   total_expected_value, expected_value_per_app
            FROM policy_run_summary
            ORDER BY run_at_utc DESC
            """,
        )

        if summary.empty:
            st.error("No policy runs found. Run: python -m src.policy")
            st.stop()

        options = summary["policy_run_id"].tolist()
        default_from_cfg = str(ctx["pipeline_cfg"].get("kpis", {}).get("policy_run_id", options[0]))
        if default_from_cfg not in options:
            default_from_cfg = options[0]

        policy_run_id = st.selectbox("Policy run", options=options, index=options.index(default_from_cfg))

        st.markdown("### Policy run summary")
        st.dataframe(summary[summary["policy_run_id"] == policy_run_id], use_container_width=True)

        # Check if KPIs exist for this policy_run_id
        funnel = _df(
            con,
            "SELECT * FROM kpi_funnel WHERE policy_run_id = ? ORDER BY decision_bucket;",
            [policy_run_id],
        )

        if funnel.empty:
            st.warning("KPIs not found for this policy run in DuckDB. Click the button below to compute them now.")

            if st.button("Compute KPIs now (execute SQL files into DuckDB)"):
                # Execute DDL files
                ddl_files = sorted(sql_ddl_dir.glob("*.sql"))
                if not ddl_files:
                    st.error(f"No DDL SQL files found in {sql_ddl_dir}")
                    st.stop()

                for f in ddl_files:
                    con.execute(f.read_text(encoding="utf-8"))

                params = {
                    "MODEL_NAME": model_name,
                    "MODEL_VERSION": model_version,
                    "POLICY_RUN_ID": policy_run_id,
                }

                kpi_files = sorted(sql_kpis_dir.glob("*.sql"))
                if not kpi_files:
                    st.error(f"No KPI SQL files found in {sql_kpis_dir}")
                    st.stop()

                for f in kpi_files:
                    raw = f.read_text(encoding="utf-8")
                    con.execute(_render_sql(raw, params))

                st.success("KPIs computed into DuckDB.")
                st.rerun()

        # Reload KPI tables for display
        funnel = _df(con, "SELECT * FROM kpi_funnel WHERE policy_run_id = ? ORDER BY decision_bucket;", [policy_run_id])
        dr = _df(
            con,
            "SELECT * FROM kpi_default_rate_by_bucket WHERE policy_run_id = ? ORDER BY decision_bucket;",
            [policy_run_id],
        )
        calib = _df(
            con,
            """
            SELECT * FROM kpi_calibration_decile
            WHERE model_name = ? AND model_version = ?
            ORDER BY decile
            """,
            [model_name, model_version],
        )
        seg = _df(
            con,
            """
            SELECT * FROM kpi_segment_risk
            WHERE model_name = ? AND model_version = ?
            ORDER BY segment_type, segment_value
            """,
            [model_name, model_version],
        )

        st.markdown("---")
        st.markdown("## Funnel (decision shares)")
        if not funnel.empty:
            fig = px.bar(funnel, x="decision_bucket", y="share", text="n")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(funnel, use_container_width=True)
        else:
            st.info("No funnel KPI rows found.")

        st.markdown("## Default rate by bucket")
        if not dr.empty:
            fig = px.bar(dr, x="decision_bucket", y="default_rate", text="n")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dr, use_container_width=True)
        else:
            st.info("No default-rate KPI rows found.")

        st.markdown("## Calibration deciles")
        if not calib.empty:
            fig = px.line(calib, x="decile", y=["avg_pd", "observed_default_rate"], markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(calib, use_container_width=True)
        else:
            st.info("No calibration KPI rows found for the selected model/version.")

        st.markdown("## Segment risk")
        if not seg.empty:
            seg_type = st.selectbox("Segment type", options=sorted(seg["segment_type"].unique().tolist()))
            seg_f = seg[seg["segment_type"] == seg_type].copy()
            fig = px.bar(seg_f, x="segment_value", y="default_rate", text="n")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(seg_f, use_container_width=True)
        else:
            st.info("No segment KPI rows found for the selected model/version.")

    finally:
        con.close()
