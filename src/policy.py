# src/policy.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import duckdb
import pandas as pd
import yaml


@dataclass(frozen=True)
class PipelineCfg:
    duckdb_path: Path


@dataclass(frozen=True)
class PolicyCfg:
    model_name: str
    model_version: str
    policy_version: str
    t_low: float
    t_high: float
    review_capacity_per_day: int
    review_approval_rate: float
    profit_good: float
    loss_default: float
    review_cost: float


def _read_pipeline_cfg(path: Path) -> PipelineCfg:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        duckdb_path = Path(cfg["paths"]["duckdb_path"])
    except Exception as e:
        raise KeyError("config/pipeline.yaml missing required key: paths.duckdb_path") from e
    return PipelineCfg(duckdb_path=duckdb_path)


def _read_policy_cfg(path: Path) -> PolicyCfg:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    try:
        model_name = str(cfg["model"]["name"])
        model_version = str(cfg["model"]["version"])

        policy_version = str(cfg["policy"]["version"])
        t_low = float(cfg["policy"]["t_low"])
        t_high = float(cfg["policy"]["t_high"])
        review_capacity_per_day = int(cfg["policy"]["review_capacity_per_day"])

        review_approval_rate = float(cfg["review"]["approval_rate"])

        profit_good = float(cfg["economics"]["profit_good"])
        loss_default = float(cfg["economics"]["loss_default"])
        review_cost = float(cfg["economics"]["review_cost"])
    except Exception as e:
        raise KeyError(
            "config/policy.yaml missing required keys. Required structure:\n"
            "model.name, model.version\n"
            "policy.version, policy.t_low, policy.t_high, policy.review_capacity_per_day\n"
            "review.approval_rate\n"
            "economics.profit_good, economics.loss_default, economics.review_cost"
        ) from e

    if not model_name or not model_version:
        raise ValueError("config/policy.yaml model.name and model.version must be non-empty strings")

    if not (0.0 <= review_approval_rate <= 1.0):
        raise ValueError("review.approval_rate must be between 0 and 1")

    if not (0.0 <= t_low < t_high <= 1.0):
        raise ValueError("t_low and t_high must satisfy 0 <= t_low < t_high <= 1")

    if review_capacity_per_day <= 0:
        raise ValueError("review_capacity_per_day must be > 0")

    return PolicyCfg(
        model_name=model_name,
        model_version=model_version,
        policy_version=policy_version,
        t_low=t_low,
        t_high=t_high,
        review_capacity_per_day=review_capacity_per_day,
        review_approval_rate=review_approval_rate,
        profit_good=profit_good,
        loss_default=loss_default,
        review_cost=review_cost,
    )


def _create_policy_tables(con: duckdb.DuckDBPyConnection) -> None:
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


def main() -> None:
    pipeline_cfg = _read_pipeline_cfg(Path("config/pipeline.yaml"))
    policy_cfg = _read_policy_cfg(Path("config/policy.yaml"))

    if not pipeline_cfg.duckdb_path.exists():
        raise FileNotFoundError(f"Missing DuckDB: {pipeline_cfg.duckdb_path}. Run: python -m src.warehouse")

    con = duckdb.connect(str(pipeline_cfg.duckdb_path))
    try:
        _create_policy_tables(con)

        df = con.execute(
            """
            SELECT s.application_id, s.pd, o.credit_risk
            FROM fact_score s
            JOIN fact_outcome o USING(application_id)
            WHERE s.model_name = ? AND s.model_version = ?
            ORDER BY s.application_id
            """,
            [policy_cfg.model_name, policy_cfg.model_version],
        ).df()

        if df.empty:
            raise RuntimeError(
                "No scores found in fact_score for the configured model_name/model_version.\n"
                "Run Phase 6 first: python -m src.train\n"
                "Then set config/policy.yaml model.name and model.version to match Phase 6 output."
            )

        df["true_default"] = (df["credit_risk"] == 2).astype(int)

        # Base decision buckets
        def bucket(pd_val: float) -> str:
            if pd_val < policy_cfg.t_low:
                return "APPROVE"
            if pd_val < policy_cfg.t_high:
                return "REVIEW"
            return "DECLINE"

        df["decision_bucket"] = df["pd"].astype(float).apply(bucket)

        # Review capacity rule (deterministic):
        # - Sort REVIEW cases by pd ascending
        # - First review_capacity_per_day are processed
        review_mask = df["decision_bucket"] == "REVIEW"
        review_df = df.loc[review_mask].sort_values("pd", ascending=True).copy()

        capacity = policy_cfg.review_capacity_per_day
        review_df["review_processed"] = 0
        review_df.iloc[:capacity, review_df.columns.get_loc("review_processed")] = 1

        # Merge processed flags back
        df["review_processed"] = 0
        df.loc[review_df.index, "review_processed"] = review_df["review_processed"]

        # Mark overflow reviews
        df.loc[(df["decision_bucket"] == "REVIEW") & (df["review_processed"] == 0), "decision_bucket"] = "REVIEW_OVERFLOW"

        # Economics (deterministic expected value):
        # - APPROVE: value = +profit_good if not default else -loss_default
        # - REVIEW processed: expected approval_rate * (approve value) - review_cost
        # - REVIEW_OVERFLOW and DECLINE: value = 0
        approve_value = (1 - df["true_default"]) * policy_cfg.profit_good - df["true_default"] * policy_cfg.loss_default

        df["expected_value"] = 0.0
        df.loc[df["decision_bucket"] == "APPROVE", "expected_value"] = approve_value[df["decision_bucket"] == "APPROVE"]

        processed_review_mask = (df["decision_bucket"] == "REVIEW") & (df["review_processed"] == 1)
        df.loc[processed_review_mask, "expected_value"] = (
            policy_cfg.review_approval_rate * approve_value[processed_review_mask] - policy_cfg.review_cost
        )

        # Summary stats
        n_total = int(len(df))
        n_approve = int((df["decision_bucket"] == "APPROVE").sum())
        n_review = int((df["decision_bucket"] == "REVIEW").sum())
        n_review_processed = int(((df["decision_bucket"] == "REVIEW") & (df["review_processed"] == 1)).sum())
        n_review_overflow = int((df["decision_bucket"] == "REVIEW_OVERFLOW").sum())
        n_decline = int((df["decision_bucket"] == "DECLINE").sum())

        if n_approve > 0:
            default_rate_approve = float(df.loc[df["decision_bucket"] == "APPROVE", "true_default"].mean())
        else:
            default_rate_approve = 0.0

        total_expected_value = float(df["expected_value"].sum())
        expected_value_per_app = float(total_expected_value / n_total)

        run_at = datetime.now(timezone.utc).isoformat()
        policy_run_id = f"{run_at.replace(':','').replace('-','')}_{policy_cfg.policy_version}_{policy_cfg.model_name}_{policy_cfg.model_version}"

        # Persist
        con.execute("DELETE FROM policy_run_summary WHERE policy_run_id = ?;", [policy_run_id])
        con.execute(
            """
            INSERT INTO policy_run_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                policy_run_id,
                run_at,
                policy_cfg.model_name,
                policy_cfg.model_version,
                policy_cfg.policy_version,
                policy_cfg.t_low,
                policy_cfg.t_high,
                policy_cfg.review_capacity_per_day,
                policy_cfg.review_approval_rate,
                policy_cfg.profit_good,
                policy_cfg.loss_default,
                policy_cfg.review_cost,
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

        print("Policy run complete")
        print(f"  - policy_run_id: {policy_run_id}")
        print("Copy this into:")
        print("  - config/pipeline.yaml -> kpis.policy_run_id")

    finally:
        con.close()


if __name__ == "__main__":
    main()
