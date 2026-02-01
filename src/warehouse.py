# src/warehouse.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import duckdb
import pandas as pd
import yaml


FEATURE_COLS = [
    "status_checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate_pct_income",
    "personal_status_and_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "people_liable_count",
    "telephone",
    "foreign_worker",
]
TARGET_COL = "credit_risk"


@dataclass(frozen=True)
class PipelineConfig:
    cache_csv: Path
    dq_report_json: Path
    ingestion_metadata_json: Path
    duckdb_path: Path


def _read_pipeline_config(path: Path) -> PipelineConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing required config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        cache_csv = Path(cfg["paths"]["cache_csv"])
        dq_report_json = Path(cfg["paths"]["dq_report_json"])
        ingestion_metadata_json = Path(cfg["paths"]["ingestion_metadata_json"])
        duckdb_path = Path(cfg["paths"]["duckdb_path"])
    except Exception as e:
        raise KeyError(
            "config/pipeline.yaml missing required keys for Phase 4: "
            "paths.cache_csv, paths.dq_report_json, paths.ingestion_metadata_json, paths.duckdb_path"
        ) from e

    return PipelineConfig(
        cache_csv=cache_csv,
        dq_report_json=dq_report_json,
        ingestion_metadata_json=ingestion_metadata_json,
        duckdb_path=duckdb_path,
    )


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_dq_passed(dq_report_path: Path) -> None:
    if not dq_report_path.exists():
        raise FileNotFoundError(
            f"Missing data quality report: {dq_report_path}. "
            f"Run Phase 3 first: python -m src.validate"
        )
    report = json.loads(dq_report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(
            f"Data quality gate FAILED. Warehouse build is blocked. "
            f"Fix validation issues and re-run Phase 3. Report: {dq_report_path}"
        )


def _load_ingestion_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ingestion metadata: {path}. Run Phase 2 first: python -m src.ingest"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _create_tables(con: duckdb.DuckDBPyConnection) -> None:
    # DuckDB persistence uses duckdb.connect(file_path). :contentReference[oaicite:19]{index=19}
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS dim_applicant (
            application_id INTEGER PRIMARY KEY,
            status_checking_account VARCHAR,
            duration_months INTEGER,
            credit_history VARCHAR,
            purpose VARCHAR,
            credit_amount INTEGER,
            savings_account_bonds VARCHAR,
            present_employment_since VARCHAR,
            installment_rate_pct_income INTEGER,
            personal_status_and_sex VARCHAR,
            other_debtors_guarantors VARCHAR,
            present_residence_since INTEGER,
            property VARCHAR,
            age_years INTEGER,
            other_installment_plans VARCHAR,
            housing VARCHAR,
            existing_credits_count INTEGER,
            job VARCHAR,
            people_liable_count INTEGER,
            telephone VARCHAR,
            foreign_worker VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_outcome (
            application_id INTEGER PRIMARY KEY,
            credit_risk INTEGER
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS meta_ingestion (
            run_id VARCHAR PRIMARY KEY,
            source VARCHAR,
            ingested_at_utc VARCHAR,
            sha256 VARCHAR,
            rows INTEGER,
            cols INTEGER,
            cache_csv VARCHAR
        );
        """
    )

    # This table is created now because later phases will insert scores into it.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_score (
            application_id INTEGER,
            model_name VARCHAR,
            model_version VARCHAR,
            pd DOUBLE,
            scored_at_utc VARCHAR,
            policy_version VARCHAR,
            decision_bucket VARCHAR
        );
        """
    )


def main() -> None:
    cfg = _read_pipeline_config(Path("config/pipeline.yaml"))

    if not cfg.cache_csv.exists():
        raise FileNotFoundError(
            f"Missing ingestion cache file: {cfg.cache_csv}. "
            f"Run Phase 2 first: python -m src.ingest"
        )

    _require_dq_passed(cfg.dq_report_json)

    _ensure_parent_dir(cfg.duckdb_path)

    df = pd.read_csv(cfg.cache_csv)

    # Create stable application_id (1..N)
    df = df.copy()
    df.insert(0, "application_id", range(1, len(df) + 1))

    dim_cols = ["application_id"] + FEATURE_COLS
    out_cols = ["application_id", TARGET_COL]

    dim_df = df[dim_cols].copy()
    out_df = df[out_cols].copy()

    meta = _load_ingestion_metadata(cfg.ingestion_metadata_json)

    # Deterministic run_id derived from ingestion metadata
    sha_prefix = str(meta.get("sha256", ""))[:12]
    retrieved_at = str(meta.get("retrieved_at_utc", ""))
    run_id = f"{retrieved_at.replace(':','').replace('-','')}_{sha_prefix}"

    con = duckdb.connect(str(cfg.duckdb_path))  # persistent DB file :contentReference[oaicite:20]{index=20}
    try:
        _create_tables(con)

        # Load into DuckDB via registered relations
        con.register("dim_df", dim_df)
        con.register("out_df", out_df)

        # Replace contents to keep build deterministic
        con.execute("DELETE FROM dim_applicant;")
        con.execute("DELETE FROM fact_outcome;")

        con.execute("INSERT INTO dim_applicant SELECT * FROM dim_df;")
        con.execute("INSERT INTO fact_outcome SELECT * FROM out_df;")

        # Upsert meta_ingestion for run_id
        con.execute("DELETE FROM meta_ingestion WHERE run_id = ?;", [run_id])
        con.execute(
            """
            INSERT INTO meta_ingestion (run_id, source, ingested_at_utc, sha256, rows, cols, cache_csv)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            [
                run_id,
                meta.get("source"),
                meta.get("retrieved_at_utc"),
                meta.get("sha256"),
                meta.get("rows"),
                meta.get("cols"),
                meta.get("cache_csv"),
            ],
        )

        # Sanity checks (hard checks, not assumptions)
        n_app = con.execute("SELECT COUNT(*) FROM dim_applicant;").fetchone()[0]
        n_out = con.execute("SELECT COUNT(*) FROM fact_outcome;").fetchone()[0]
        if n_app != n_out:
            raise RuntimeError(f"Row mismatch: dim_applicant={n_app}, fact_outcome={n_out}")

        print("✅ Phase 4 complete")
        print(f"  - DuckDB: {cfg.duckdb_path}")
        print(f"  - dim_applicant rows: {n_app}")
        print(f"  - fact_outcome rows: {n_out}")
        print(f"  - meta_ingestion run_id: {run_id}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
