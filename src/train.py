# src/train.py
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from src.features import logistic_coefficients_table, get_feature_names_from_preprocessor
from src.metrics import compute_metrics, save_json
from src.models.logreg import train_logreg
from src.models.lightgbm import train_lightgbm
import yaml


NUMERIC_COLS = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct_income",
    "present_residence_since",
    "age_years",
    "existing_credits_count",
    "people_liable_count",
]

CATEGORICAL_COLS = [
    "status_checking_account",
    "credit_history",
    "purpose",
    "savings_account_bonds",
    "present_employment_since",
    "personal_status_and_sex",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


@dataclass(frozen=True)
class TrainConfig:
    duckdb_path: Path
    dq_report_json: Path
    models_dir: Path
    random_seed: int


def _read_pipeline_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    try:
        duckdb_path = Path(cfg["paths"]["duckdb_path"])
        dq_report_json = Path(cfg["paths"]["dq_report_json"])
        models_dir = Path(cfg["paths"]["models_dir"])
        random_seed = int(cfg["run"]["random_seed"])
    except Exception as e:
        raise KeyError(
            "config/pipeline.yaml missing required keys for Phase 6: "
            "paths.duckdb_path, paths.dq_report_json, paths.models_dir, run.random_seed"
        ) from e

    return TrainConfig(
        duckdb_path=duckdb_path,
        dq_report_json=dq_report_json,
        models_dir=models_dir,
        random_seed=random_seed,
    )


def _require_dq_passed(dq_report_json: Path) -> None:
    if not dq_report_json.exists():
        raise FileNotFoundError(f"Missing DQ report: {dq_report_json}. Run: python -m src.validate")

    report = json.loads(dq_report_json.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(f"DQ gate failed. Training blocked. Fix DQ and re-run validate. Report: {dq_report_json}")


def _git_short_sha() -> str:
    # Strict: must run inside a git repo with git installed.
    out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    if not out:
        raise RuntimeError("Could not read git short SHA")
    return out


def _make_model_version() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sha = _git_short_sha()
    return f"{ts}_{sha}"


def _load_training_data(con: duckdb.DuckDBPyConnection) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = con.execute(
        """
        SELECT a.*, o.credit_risk
        FROM dim_applicant a
        JOIN fact_outcome o USING(application_id)
        ORDER BY application_id
        """
    ).df()

    # y: default = 1 if credit_risk == 2 (bad), else 0
    y = (df["credit_risk"] == 2).astype(int)
    y.name = "default"

    X = df[FEATURE_COLS].copy()

    # Ensure categoricals are strings for OneHotEncoder
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(str)

    return X, y, df[["application_id"]].copy()


def _lgbm_feature_importance_table(pipeline) -> pd.DataFrame:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    feat_names = get_feature_names_from_preprocessor(preprocess)
    importances = model.feature_importances_  # LightGBM exposes feature_importances_ :contentReference[oaicite:9]{index=9}

    if len(feat_names) != len(importances):
        # Hard fail to keep traceability strict
        raise RuntimeError("Feature name length does not match feature_importances_ length")

    out = pd.DataFrame({"feature": feat_names, "importance": importances})
    out["importance"] = out["importance"].astype(float)
    out = out.sort_values("importance", ascending=False)
    return out


def _write_scores_to_duckdb(
    con: duckdb.DuckDBPyConnection,
    application_ids: pd.Series,
    pd_scores: pd.Series,
    model_name: str,
    model_version: str,
) -> None:
    scored_at = datetime.now(timezone.utc).isoformat()

    score_df = pd.DataFrame(
        {
            "application_id": application_ids.astype(int),
            "model_name": model_name,
            "model_version": model_version,
            "pd": pd_scores.astype(float),
            "scored_at_utc": scored_at,
            "policy_version": None,
            "decision_bucket": None,
        }
    )

    con.register("score_df", score_df)
    con.execute(
        "DELETE FROM fact_score WHERE model_name = ? AND model_version = ?;",
        [model_name, model_version],
    )
    con.execute(
        """
        INSERT INTO fact_score
        SELECT application_id, model_name, model_version, pd, scored_at_utc, policy_version, decision_bucket
        FROM score_df;
        """
    )


def _export_model_bundle(
    cfg: TrainConfig,
    model_name: str,
    model_version: str,
    pipeline,
    metrics: Dict[str, Any],
    test_pred_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> Path:
    out_dir = cfg.models_dir / model_name / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    dump(pipeline, out_dir / "model.joblib")
    save_json(metrics, out_dir / "metrics.json")
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)

    return out_dir


def main() -> None:
    cfg = _read_pipeline_config(Path("config/pipeline.yaml"))
    _require_dq_passed(cfg.dq_report_json)

    if not cfg.duckdb_path.exists():
        raise FileNotFoundError(f"Missing DuckDB: {cfg.duckdb_path}. Run: python -m src.warehouse")

    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        X, y, app_ids_df = _load_training_data(con)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X,
            y,
            app_ids_df["application_id"],
            test_size=0.2,
            stratify=y,
            random_state=cfg.random_seed,
        )

        # --------- Train LogReg ----------
        logreg_version = _make_model_version()
        logreg_res = train_logreg(
            X_train=X_train,
            y_train=y_train,
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            random_state=cfg.random_seed,
        )
        logreg_pipe = logreg_res.pipeline

        logreg_test_proba = logreg_pipe.predict_proba(X_test)[:, 1]
        logreg_metrics = compute_metrics(y_test, logreg_test_proba, thresholds=(0.2, 0.35, 0.5))
        logreg_metrics["best_params"] = logreg_res.best_params
        logreg_metrics["model_name"] = "logreg"
        logreg_metrics["model_version"] = logreg_version

        logreg_test_pred_df = pd.DataFrame(
            {"application_id": ids_test.values, "y_true": y_test.values, "pd_default": logreg_test_proba}
        )

        logreg_importance_df = logistic_coefficients_table(logreg_pipe)

        # Score full dataset and write to DuckDB
        logreg_full_proba = logreg_pipe.predict_proba(X)[:, 1]
        _write_scores_to_duckdb(
            con,
            application_ids=app_ids_df["application_id"],
            pd_scores=pd.Series(logreg_full_proba),
            model_name="logreg",
            model_version=logreg_version,
        )

        _export_model_bundle(
            cfg,
            model_name="logreg",
            model_version=logreg_version,
            pipeline=logreg_pipe,
            metrics=logreg_metrics,
            test_pred_df=logreg_test_pred_df,
            importance_df=logreg_importance_df,
        )

        # --------- Train LightGBM ----------
        lgbm_version = _make_model_version()
        lgbm_res = train_lightgbm(
            X_train=X_train,
            y_train=y_train,
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            random_state=cfg.random_seed,
        )
        lgbm_pipe = lgbm_res.pipeline

        lgbm_test_proba = lgbm_pipe.predict_proba(X_test)[:, 1]  # predict_proba supported :contentReference[oaicite:10]{index=10}
        lgbm_metrics = compute_metrics(y_test, lgbm_test_proba, thresholds=(0.2, 0.35, 0.5))
        lgbm_metrics["best_params"] = lgbm_res.best_params
        lgbm_metrics["model_name"] = "lightgbm"
        lgbm_metrics["model_version"] = lgbm_version

        lgbm_test_pred_df = pd.DataFrame(
            {"application_id": ids_test.values, "y_true": y_test.values, "pd_default": lgbm_test_proba}
        )

        lgbm_importance_df = _lgbm_feature_importance_table(lgbm_pipe)

        lgbm_full_proba = lgbm_pipe.predict_proba(X)[:, 1]
        _write_scores_to_duckdb(
            con,
            application_ids=app_ids_df["application_id"],
            pd_scores=pd.Series(lgbm_full_proba),
            model_name="lightgbm",
            model_version=lgbm_version,
        )

        _export_model_bundle(
            cfg,
            model_name="lightgbm",
            model_version=lgbm_version,
            pipeline=lgbm_pipe,
            metrics=lgbm_metrics,
            test_pred_df=lgbm_test_pred_df,
            importance_df=lgbm_importance_df,
        )

        print("Training complete")
        print(f"  - logreg model_version: {logreg_version}")
        print(f"  - lightgbm model_version: {lgbm_version}")
        print("Copy one of these into:")
        print("  - config/policy.yaml -> model.version")
        print("  - config/pipeline.yaml -> kpis.model_version (later)")

    finally:
        con.close()


if __name__ == "__main__":
    main()
