# src/kpis.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import duckdb
import yaml


@dataclass(frozen=True)
class KPIConfig:
    duckdb_path: Path
    sql_ddl_dir: Path
    sql_kpis_dir: Path
    kpis_export_dir: Path
    model_name: str
    model_version: str
    policy_run_id: str


def _read_kpi_config(path: Path) -> KPIConfig:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        duckdb_path = Path(cfg["paths"]["duckdb_path"])
        sql_ddl_dir = Path(cfg["paths"]["sql_ddl_dir"])
        sql_kpis_dir = Path(cfg["paths"]["sql_kpis_dir"])
        kpis_export_dir = Path(cfg["paths"]["kpis_export_dir"])
        model_name = str(cfg["kpis"]["model_name"])
        model_version = str(cfg["kpis"]["model_version"])
        policy_run_id = str(cfg["kpis"]["policy_run_id"])
    except Exception as e:
        raise KeyError(
            "config/pipeline.yaml missing required keys for Phase 5:\n"
            "paths.duckdb_path, paths.sql_ddl_dir, paths.sql_kpis_dir, paths.kpis_export_dir\n"
            "kpis.model_name, kpis.model_version, kpis.policy_run_id"
        ) from e

    if not model_name or not model_version or not policy_run_id:
        raise ValueError(
            "You must fill these in config/pipeline.yaml before running KPIs:\n"
            "kpis.model_name, kpis.model_version, kpis.policy_run_id"
        )

    return KPIConfig(
        duckdb_path=duckdb_path,
        sql_ddl_dir=sql_ddl_dir,
        sql_kpis_dir=sql_kpis_dir,
        kpis_export_dir=kpis_export_dir,
        model_name=model_name,
        model_version=model_version,
        policy_run_id=policy_run_id,
    )


def _render_sql(sql: str, params: Dict[str, str]) -> str:
    for k, v in params.items():
        sql = sql.replace(f"{{{{{k}}}}}", v)
    return sql


def main() -> None:
    cfg = _read_kpi_config(Path("config/pipeline.yaml"))

    if not cfg.duckdb_path.exists():
        raise FileNotFoundError(f"Missing DuckDB: {cfg.duckdb_path}. Run: python -m src.warehouse")

    # Create export directory for this KPI run
    run_at = datetime.now(timezone.utc).isoformat()
    kpi_run_id = f"{run_at.replace(':','').replace('-','')}_{cfg.policy_run_id}"
    export_dir = cfg.kpis_export_dir / kpi_run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        # Execute DDL
        ddl_files = sorted(cfg.sql_ddl_dir.glob("*.sql"))
        if not ddl_files:
            raise FileNotFoundError(f"No DDL SQL files found in {cfg.sql_ddl_dir}")

        for f in ddl_files:
            con.execute(f.read_text(encoding="utf-8"))

        params = {
            "MODEL_NAME": cfg.model_name,
            "MODEL_VERSION": cfg.model_version,
            "POLICY_RUN_ID": cfg.policy_run_id,
        }

        # Execute KPI SQL scripts
        kpi_files = sorted(cfg.sql_kpis_dir.glob("*.sql"))
        if not kpi_files:
            raise FileNotFoundError(f"No KPI SQL files found in {cfg.sql_kpis_dir}")

        for f in kpi_files:
            raw = f.read_text(encoding="utf-8")
            rendered = _render_sql(raw, params)
            con.execute(rendered)

        # Export KPI tables filtered to current params
        funnel = con.execute(
            "SELECT * FROM kpi_funnel WHERE policy_run_id = ? ORDER BY decision_bucket;",
            [cfg.policy_run_id],
        ).df()
        funnel.to_csv(export_dir / "kpi_funnel.csv", index=False)

        dr = con.execute(
            "SELECT * FROM kpi_default_rate_by_bucket WHERE policy_run_id = ? ORDER BY decision_bucket;",
            [cfg.policy_run_id],
        ).df()
        dr.to_csv(export_dir / "kpi_default_rate_by_bucket.csv", index=False)

        calib = con.execute(
            """
            SELECT * FROM kpi_calibration_decile
            WHERE model_name = ? AND model_version = ?
            ORDER BY decile;
            """,
            [cfg.model_name, cfg.model_version],
        ).df()
        calib.to_csv(export_dir / "kpi_calibration_decile.csv", index=False)

        seg = con.execute(
            """
            SELECT * FROM kpi_segment_risk
            WHERE model_name = ? AND model_version = ?
            ORDER BY segment_type, segment_value;
            """,
            [cfg.model_name, cfg.model_version],
        ).df()
        seg.to_csv(export_dir / "kpi_segment_risk.csv", index=False)

        manifest = {
            "kpi_run_id": kpi_run_id,
            "duckdb_path": str(cfg.duckdb_path),
            "model_name": cfg.model_name,
            "model_version": cfg.model_version,
            "policy_run_id": cfg.policy_run_id,
            "exports": [
                str(export_dir / "kpi_funnel.csv"),
                str(export_dir / "kpi_default_rate_by_bucket.csv"),
                str(export_dir / "kpi_calibration_decile.csv"),
                str(export_dir / "kpi_segment_risk.csv"),
            ],
        }
        (export_dir / "manifest.json").write_text(yaml.safe_dump(manifest), encoding="utf-8")

        print("Version-controlled SQL KPI layer complete")
        print(f"  - KPI export dir: {export_dir}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
