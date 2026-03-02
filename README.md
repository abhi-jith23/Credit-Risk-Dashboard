# Credit Risk Dashboard (v2)

This repository contains a reproducible credit-risk decisioning workflow built around the German Credit (UCI Statlog) dataset. It combines a lightweight Streamlit dashboard with an end-to-end pipeline that ingests data, validates it, loads it into a DuckDB warehouse, trains two models (Logistic Regression and LightGBM) with versioned artifacts, simulates underwriting policies, and produces BI-ready KPIs defined in version-controlled SQL.

## Project Overview

The project is organized as a BI + ML pipeline rather than only a classifier. It produces:
- Model scores (Probability of Default) for applicants
- Governance-style model diagnostics (ROC, PR, calibration, threshold analysis)
- A persistent analytics warehouse (DuckDB) that stores applicants, outcomes, scores, policy decisions, and KPI tables
- A policy simulator that turns scores into business actions (Approve / Review / Decline) under capacity and economics assumptions
- A SQL KPI layer to compute portfolio and decision KPIs in a reproducible and auditable way
- A Streamlit app that reads from versioned model artifacts and DuckDB

## Key Features

Modeling and scoring
- Two models trained on the same validated dataset
  - Logistic Regression (interpretable baseline)
  - LightGBM (nonlinear challenger)
- Versioned model artifacts stored under `artifacts/models/<model>/<model_version>/`
- Model scores written into DuckDB `fact_score` with `model_name` and `model_version`

Data integrity and reproducibility
- Deterministic ingestion cache and lineage metadata (hash + timestamp)
- Data quality contract with a fail-closed gate (validation report is produced every run)
- Pinned dependencies in `requirements.txt`
- Persistent DuckDB warehouse used as the single source of truth for analytics

Decisioning and business impact
- Policy simulator converts PD into underwriting actions:
  - PD < t_low → Approve
  - t_low ≤ PD < t_high → Review
  - PD ≥ t_high → Decline
- Review capacity constraint and overflow tracking
- Economics layer (profit/loss assumptions and review cost)
- Results written into DuckDB policy tables for BI and dashboard use

BI and metrics
- KPI definitions written as version-controlled SQL in `sql/kpis/`
- KPI tables materialized into DuckDB and exported as CSV snapshots per run
- Metrics cover funnel rates, default rates by decision, calibration deciles, and segment risk

## Reproducibility

The project is artifact-driven and warehouse-driven:
- Inputs are cached and fingerprinted:
  - `data/cache/german_credit_raw.csv`
  - `artifacts/ingestion/ingestion_metadata.json` (includes SHA256 and retrieval timestamp)
- Data quality is enforced before downstream steps:
  - `artifacts/data_quality/data_quality_report.json`
- Models are never overwritten; each training produces a new version folder:
  - `artifacts/models/logreg/<model_version>/...`
  - `artifacts/models/lightgbm/<model_version>/...`
- KPIs are reproducible and auditable via SQL files committed to git:
  - `sql/kpis/*.sql`
  - KPI exports are saved per run under `artifacts/kpis/<kpi_run_id>/`

## Project Structure

```text
.
├── app
│   ├── main_app.py
│   ├── ui_text.py
│   └── pages
│       ├── applicant_scoring.py
│       ├── portfolio_explorer.py
│       ├── model_card.py
│       ├── data_quality.py
│       ├── bi_metrics.py
│       └── policy_impact.py
├── artifacts
│   ├── ingestion
│   │   └── ingestion_metadata.json
│   ├── data_quality
│   │   └── data_quality_report.json
│   ├── models
│   │   ├── logreg
│   │   │   └── <model_version>/
│   │   └── lightgbm
│   │       └── <model_version>/
│   └── kpis
│       └── <kpi_run_id>/
├── config
│   ├── pipeline.yaml
│   └── policy.yaml
├── data
│   ├── cache
│   │   └── german_credit_raw.csv
│   └── german.data
├── sql
│   ├── ddl
│   │   └── 001_create_tables.sql
│   └── kpis
│       ├── kpi_funnel.sql
│       ├── kpi_default_rate.sql
│       ├── kpi_calibration_decile.sql
│       └── kpi_segment_risk.sql
├── src
│   ├── ingest.py
│   ├── validate.py
│   ├── warehouse.py
│   ├── train.py
│   ├── policy.py
│   ├── kpis.py
│   ├── pipeline.py
│   └── models
│       ├── logreg.py
│       └── lightgbm.py
├── warehouse
│   └── credit_risk.duckdb
├── requirements.txt
└── README.md
```

## Data Flow

This section describes what was implemented in v2 and why each step exists. It also explains how to replicate the same pipeline for a different dataset.

Ingest
- What happens:
  - The dataset is retrieved programmatically and saved into `data/cache/german_credit_raw.csv`.
  - A lineage metadata file is written to `artifacts/ingestion/ingestion_metadata.json` (timestamp, source ID, file hash, row/col counts).
- Why it is needed:
  - It makes the input data reproducible, traceable, and easy to refresh.
- How to replicate with your own dataset:
  - Replace the ingestion step to write your own raw file to `data/cache/<your_data>.csv`.
  - Update the ingestion metadata to include your file hash and retrieval timestamp.
  - Keep the downstream steps unchanged as long as your schema matches the validation and warehouse expectations.

Validate (data contract)
- What happens:
  - A strict data contract validates schema, types, allowed categories, and numeric ranges.
  - A machine-readable report is written to `artifacts/data_quality/data_quality_report.json`.
  - Downstream steps are blocked if validation fails.
- Why it is needed:
  - It prevents bad or unexpected data from entering the warehouse, models, and KPI computations.
- How to replicate with your own dataset:
  - Define your dataset schema (column names and types).
  - Define allowed categories for categorical fields and range checks for numeric fields.
  - Ensure the validation report is generated for every run and that failures stop the pipeline.

Warehouse (DuckDB)
- What happens:
  - A persistent DuckDB database is created at `warehouse/credit_risk.duckdb`.
  - Clean tables are created and loaded:
    - `dim_applicant` (features)
    - `fact_outcome` (target/outcome)
    - `fact_score` (model scores, model versions)
    - `meta_ingestion` (ingestion lineage)
- Why it is needed:
  - The warehouse becomes the single source of truth for BI queries, policy simulation, and dashboard metrics.
  - SQL KPIs and monitoring can be computed consistently on stored tables.
- How to replicate with your own dataset:
  - Create a `dim_*` table for stable features and a `fact_*` table for outcomes.
  - Ensure you have a stable key (`application_id`) so policy decisions and scores can be joined.

Train (two-model system)
- What happens:
  - Both Logistic Regression and LightGBM are trained from warehouse tables.
  - Each run generates a deterministic `model_version` and stores a self-contained bundle:
    - `model.joblib`, `metrics.json`, `test_predictions.csv`, `feature_importance.csv`
  - Scores for the full dataset are inserted into DuckDB `fact_score`.
- Why it is needed:
  - Versioned artifacts support traceability, comparison between models, and reproducible inference.
  - A champion–challenger setup creates a realistic BI + ML workflow.
- How to replicate with your own dataset:
  - Ensure the warehouse outputs the same types of training inputs (features + label).
  - Keep the artifact layout and write scores back into the warehouse with model identifiers.

Policy (impact simulator)
- What happens:
  - A policy run reads PD scores from `fact_score` for a chosen model/version.
  - It applies thresholds and review capacity and computes expected value using economics parameters.
  - It writes:
    - `policy_run_summary`
    - `policy_run_decisions`
  - A `policy_run_id` uniquely identifies each run.
- Why it is needed:
  - It converts “a score” into “a business decision” and allows operational constraints to be modeled.
- How to replicate with your own dataset:
  - Keep the same policy logic if you have PD scores and an outcome label.
  - Change the economics assumptions to match your business case.

KPIs (SQL KPI layer)
- What happens:
  - KPI definitions are stored as SQL files in `sql/kpis/`.
  - Running the KPI step materializes tables in DuckDB and exports CSV snapshots:
    - `kpi_funnel`
    - `kpi_default_rate_by_bucket`
    - `kpi_calibration_decile`
    - `kpi_segment_risk`
  - Exports are written to `artifacts/kpis/<kpi_run_id>/`.
- Why it is needed:
  - SQL KPIs are auditable, version-controlled, and repeatable.
  - They directly support BI-style reporting and segmentation.
- How to replicate with your own dataset:
  - Define KPIs relevant to your business in SQL files.
  - Ensure the KPI queries reference your warehouse tables consistently.

## BI and SQL Capabilities

DuckDB is used as the analytics core:
- Persistent tables for applicants, outcomes, model scores, and policy runs
- KPI tables computed using version-controlled SQL
- Easy segmentation and monitoring via SQL joins and aggregates

KPI definitions are maintained as code:
- DDL: `sql/ddl/001_create_tables.sql`
- KPIs: `sql/kpis/*.sql`
- KPI outputs:
  - materialized tables in DuckDB
  - exported CSV snapshots under `artifacts/kpis/<kpi_run_id>/`

## Setup and Usage

Install dependencies
```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard
```bash
streamlit run app/main_app.py
```

Run the pipeline step-by-step
```bash
python -m src.ingest
python -m src.validate
python -m src.warehouse
python -m src.train
python -m src.policy
python -m src.kpis
```

Run the end-to-end pipeline
```bash
python -m src.pipeline
```

## Key Improvements in the Next Update

Integrated dataset ingestion and refresh without manual configuration changes
- Add an ingestion mode that accepts user-provided files (CSV or similar) as the source.
- Automatically rebuild the warehouse and retrain models whenever a new file is ingested.
- Automatically select the latest model version and latest policy run for KPI computation.
- Automatically update KPI tables and exports after retraining.

## Disclaimer

This project is an educational demonstration of a credit-risk workflow. It is not a production credit-scoring system and must not be used for real lending decisions, credit approvals, or compliance judgments. Real-world systems require additional validation, governance, monitoring, and fairness assessments.

## Author

Abhijith Senthilkumar  
MSc Data Science, University of Luxembourg  
GitHub: https://github.com/abhi-jith23  
Email: abhijith.unilu@gmail.com