# src/ingest.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from ucimlrepo import fetch_ucirepo


# Canonical column names based on UCI variable information for german.data (20 attributes + target)
# UCI provides the variable list and categorical code sets. :contentReference[oaicite:4]{index=4}
FEATURE_COLUMNS = [
    "status_checking_account",                 # Attribute 1 (A11..A14)
    "duration_months",                         # Attribute 2
    "credit_history",                          # Attribute 3 (A30..A34)
    "purpose",                                 # Attribute 4 (A40..A410)
    "credit_amount",                           # Attribute 5
    "savings_account_bonds",                   # Attribute 6 (A61..A65)
    "present_employment_since",                # Attribute 7 (A71..A75)
    "installment_rate_pct_income",             # Attribute 8
    "personal_status_and_sex",                 # Attribute 9 (A91..A95)
    "other_debtors_guarantors",                # Attribute 10 (A101..A103)
    "present_residence_since",                 # Attribute 11
    "property",                                # Attribute 12 (A121..A124)
    "age_years",                               # Attribute 13
    "other_installment_plans",                 # Attribute 14 (A141..A143)
    "housing",                                 # Attribute 15 (A151..A153)
    "existing_credits_count",                  # Attribute 16
    "job",                                     # Attribute 17 (A171..A174)
    "people_liable_count",                     # Attribute 18
    "telephone",                               # Attribute 19 (A191..A192)
    "foreign_worker",                          # Attribute 20 (A201..A202)
]
TARGET_COLUMN = "credit_risk"                  # 1 = Good, 2 = Bad :contentReference[oaicite:5]{index=5}


@dataclass(frozen=True)
class PipelineConfig:
    uci_id: int
    cache_csv: Path
    ingestion_metadata_json: Path
    random_seed: int


def _read_pipeline_config(path: Path) -> PipelineConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required config file: {path}. "
            f"Create it with keys: dataset.uci_id, paths.cache_csv, paths.ingestion_metadata_json, run.random_seed."
        )

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Strict required keys (no assumptions)
    try:
        uci_id = int(cfg["dataset"]["uci_id"])
        cache_csv = Path(cfg["paths"]["cache_csv"])
        ingestion_metadata_json = Path(cfg["paths"]["ingestion_metadata_json"])
        random_seed = int(cfg["run"]["random_seed"])
    except Exception as e:
        raise KeyError(
            "config/pipeline.yaml is missing required keys or has invalid types. "
            "Required: dataset.uci_id (int), paths.cache_csv (str), "
            "paths.ingestion_metadata_json (str), run.random_seed (int)."
        ) from e

    return PipelineConfig(
        uci_id=uci_id,
        cache_csv=cache_csv,
        ingestion_metadata_json=ingestion_metadata_json,
        random_seed=random_seed,
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fetch_uci_144_as_dataframe(uci_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    ds = fetch_ucirepo(id=uci_id)  # UCI shows this exact usage for id=144. :contentReference[oaicite:6]{index=6}
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    # y may be a DataFrame with one column; normalize to Series
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"Expected exactly 1 target column, got {y.shape[1]}")
        y_series = y.iloc[:, 0]
    else:
        y_series = pd.Series(y)

    return X, y_series


def main() -> None:
    cfg = _read_pipeline_config(Path("config/pipeline.yaml"))

    _ensure_parent_dir(cfg.cache_csv)
    _ensure_parent_dir(cfg.ingestion_metadata_json)

    # Fetch from UCI using ucimlrepo, per UCI documentation. :contentReference[oaicite:7]{index=7}
    X, y = _fetch_uci_144_as_dataframe(cfg.uci_id)

    # Enforce column count and rename to canonical names for stable downstream validation/warehouse.
    if X.shape[1] != 20:
        raise ValueError(f"Expected 20 feature columns, got {X.shape[1]}. UCI states 20 features.")

    X.columns = FEATURE_COLUMNS
    df = X.copy()
    df[TARGET_COLUMN] = y.values

    # Write cache CSV deterministically (no index column)
    df.to_csv(cfg.cache_csv, index=False)

    sha256 = _sha256_file(cfg.cache_csv)
    now_utc = datetime.now(timezone.utc).isoformat()

    meta: Dict[str, Any] = {
        "retrieved_at_utc": now_utc,
        "source": f"uci:{cfg.uci_id}",
        "cache_csv": str(cfg.cache_csv),
        "sha256": sha256,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "feature_cols": FEATURE_COLUMNS,
        "target_col": TARGET_COLUMN,
    }

    with cfg.ingestion_metadata_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Phase 2 complete")
    print(f"  - Wrote: {cfg.cache_csv}")
    print(f"  - Wrote: {cfg.ingestion_metadata_json}")
    print(f"  - sha256: {sha256}")


if __name__ == "__main__":
    main()
