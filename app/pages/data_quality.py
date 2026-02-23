# app/pages/data_quality.py
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render(ctx: dict) -> None:
    st.subheader("Data Quality")

    pipeline_cfg = ctx["pipeline_cfg"]
    project_root: Path = ctx["project_root"]

    dq_path = project_root / Path(pipeline_cfg["paths"]["dq_report_json"])
    ingestion_path = project_root / Path(pipeline_cfg["paths"]["ingestion_metadata_json"])

    dq = _read_json(dq_path)
    ing = _read_json(ingestion_path)

    if not dq:
        st.error(f"Missing DQ report: {dq_path}")
        st.stop()

    passed = dq.get("passed")
    if passed is True:
        st.success("DQ Gate: PASSED")
    elif passed is False:
        st.error("DQ Gate: FAILED")
    else:
        st.warning("DQ Gate: Status unknown (report does not contain 'passed').")

    st.markdown("### Ingestion metadata")
    if ing:
        st.json(ing)
    else:
        st.warning(f"Missing ingestion metadata: {ingestion_path}")

    st.markdown("### Data quality report")
    st.json(dq)
