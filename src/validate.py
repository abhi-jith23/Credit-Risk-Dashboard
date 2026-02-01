# src/validate.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
import yaml


# UCI categorical code sets are listed in "Additional Variable Information". :contentReference[oaicite:11]{index=11}
ALLOWED = {
    "status_checking_account": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
    "savings_account_bonds": ["A61", "A62", "A63", "A64", "A65"],
    "present_employment_since": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status_and_sex": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors_guarantors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

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
TARGET_COL = "credit_risk"  # 1=Good, 2=Bad :contentReference[oaicite:12]{index=12}


@dataclass(frozen=True)
class PipelineConfig:
    cache_csv: Path
    dq_report_json: Path


def _read_pipeline_config(path: Path) -> PipelineConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing required config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        cache_csv = Path(cfg["paths"]["cache_csv"])
        dq_report_json = Path(cfg["paths"]["dq_report_json"])
    except Exception as e:
        raise KeyError(
            "config/pipeline.yaml missing required keys for Phase 3: "
            "paths.cache_csv, paths.dq_report_json"
        ) from e

    return PipelineConfig(cache_csv=cache_csv, dq_report_json=dq_report_json)


def _build_schema() -> DataFrameSchema:
    # Use lazy=True during validate to collect all errors. :contentReference[oaicite:13]{index=13}
    # Use coerce=True on columns to coerce types before checks. :contentReference[oaicite:14]{index=14}
    schema = DataFrameSchema(
        {
            # Categorical columns
            "status_checking_account": Column(str, Check.isin(ALLOWED["status_checking_account"]), nullable=False, coerce=True),
            "credit_history": Column(str, Check.isin(ALLOWED["credit_history"]), nullable=False, coerce=True),
            "purpose": Column(str, Check.isin(ALLOWED["purpose"]), nullable=False, coerce=True),
            "savings_account_bonds": Column(str, Check.isin(ALLOWED["savings_account_bonds"]), nullable=False, coerce=True),
            "present_employment_since": Column(str, Check.isin(ALLOWED["present_employment_since"]), nullable=False, coerce=True),
            "personal_status_and_sex": Column(str, Check.isin(ALLOWED["personal_status_and_sex"]), nullable=False, coerce=True),
            "other_debtors_guarantors": Column(str, Check.isin(ALLOWED["other_debtors_guarantors"]), nullable=False, coerce=True),
            "property": Column(str, Check.isin(ALLOWED["property"]), nullable=False, coerce=True),
            "other_installment_plans": Column(str, Check.isin(ALLOWED["other_installment_plans"]), nullable=False, coerce=True),
            "housing": Column(str, Check.isin(ALLOWED["housing"]), nullable=False, coerce=True),
            "job": Column(str, Check.isin(ALLOWED["job"]), nullable=False, coerce=True),
            "telephone": Column(str, Check.isin(ALLOWED["telephone"]), nullable=False, coerce=True),
            "foreign_worker": Column(str, Check.isin(ALLOWED["foreign_worker"]), nullable=False, coerce=True),

            # Numeric columns (UCI: integers; and no missing values) :contentReference[oaicite:15]{index=15}
            "duration_months": Column(int, Check.gt(0), nullable=False, coerce=True),
            "credit_amount": Column(int, Check.gt(0), nullable=False, coerce=True),
            "installment_rate_pct_income": Column(int, Check.isin([1, 2, 3, 4]), nullable=False, coerce=True),
            "present_residence_since": Column(int, Check.isin([1, 2, 3, 4]), nullable=False, coerce=True),
            "age_years": Column(int, Check.gt(0), nullable=False, coerce=True),
            "existing_credits_count": Column(int, Check.gt(0), nullable=False, coerce=True),
            "people_liable_count": Column(int, Check.isin([1, 2]), nullable=False, coerce=True),

            # Target
            TARGET_COL: Column(int, Check.isin([1, 2]), nullable=False, coerce=True),
        },
        strict=True,  # reject extra columns so contract stays explicit
    )
    return schema


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_failure_cases(exc: pa.errors.SchemaErrors) -> List[Dict[str, Any]]:
    # Pandera SchemaErrors contains failure_cases when lazy=True. :contentReference[oaicite:16]{index=16}
    fc = exc.failure_cases.copy()
    # Keep only common fields if present
    keep = [c for c in ["schema_context", "column", "check", "failure_case", "index"] if c in fc.columns]
    fc = fc[keep] if keep else fc
    return fc.to_dict(orient="records")


def main() -> None:
    cfg = _read_pipeline_config(Path("config/pipeline.yaml"))
    _ensure_parent_dir(cfg.dq_report_json)

    if not cfg.cache_csv.exists():
        raise FileNotFoundError(
            f"Missing ingestion cache file: {cfg.cache_csv}. "
            f"Run Phase 2 first: python -m src.ingest"
        )

    df = pd.read_csv(cfg.cache_csv)

    report: Dict[str, Any] = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(cfg.cache_csv),
        "passed": False,
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
        "errors": [],
    }

    schema = _build_schema()

    try:
        schema.validate(df, lazy=True)  # lazy validation collects all issues :contentReference[oaicite:17]{index=17}
        report["passed"] = True
    except pa.errors.SchemaErrors as exc:
        report["passed"] = False
        report["errors"] = _format_failure_cases(exc)

    with cfg.dq_report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if report["passed"]:
        print("✅ Phase 3 complete: Data Quality PASSED")
    else:
        print("❌ Phase 3 complete: Data Quality FAILED")
        print(f"See report: {cfg.dq_report_json}")


if __name__ == "__main__":
    main()
