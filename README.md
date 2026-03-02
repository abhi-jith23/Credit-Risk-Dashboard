# Credit Risk Scoring Dashboard (v2)

An interactive **credit risk scoring** dashboard that predicts **Probability of Default (PD)** for loan applicants using **two trained models** (**Logistic Regression** and **LightGBM**) with a reproducible, end-to-end pipeline.
The app is designed like a lightweight internal tool used in lending teams: score an applicant, explore a portfolio, review a governance-style model card, inspect data quality, run underwriting policies, and view BI KPIs computed in SQL.

---

## Project Overview

This project builds a production-style workflow on the **Statlog German Credit** dataset and serves it through a **Streamlit** web application backed by a **DuckDB warehouse**.

What you can do with it:
- **Underwriting view:** enter applicant details → get PD + a decision suggestion (Approve / Review / Decline).
- **Portfolio view:** filter the dataset and inspect PD distributions and segmentation.
- **Model governance view:** inspect ROC/PR curves, calibration, confusion matrices at different thresholds, and feature importance.
- **Data quality view:** see the validation report that gates downstream steps.
- **Policy impact view:** simulate a decision policy with thresholds, review capacity, and economics.
- **BI metrics view:** view KPIs computed from version-controlled SQL and stored in DuckDB.

---

## Key Features

- **Two-model system** with versioned artifacts:
  - Logistic Regression (interpretable baseline)
  - LightGBM (nonlinear challenger)
- **Model selector + model version selector** in Streamlit (reads versioned artifacts from `artifacts/models/`)
- **DuckDB warehouse** as a persistent analytics core (`warehouse/credit_risk.duckdb`)
- **Data quality gate** with machine-readable report (`artifacts/data_quality/data_quality_report.json`)
- **Policy simulator** (Approve / Review / Decline) with:
  - two-threshold decisioning
  - review capacity constraint + overflow
  - economics layer (profit/loss assumptions, review cost)
  - results stored in DuckDB policy tables
- **BI KPI layer in SQL**:
  - KPI definitions stored in `sql/kpis/`
  - KPI tables materialized in DuckDB and exported as per-run CSV snapshots

---

## Installation (quick start)

### Docker Method (recommended)

**What you need installed:**
- **Docker**
  - **Windows:** Docker Desktop with **WSL 2 enabled**
  - **macOS:** Docker Desktop
  - **Linux:** Docker Engine or Docker Desktop

```bash
# 1) Clone the repo and navigate into it
git clone https://github.com/abhi-jith23/Credit-Risk-Dashboard.git
cd Credit-Risk-Dashboard

# 2) Build the image (run from the project root where the Dockerfile is)
docker build -t credit-risk-dashboard .

# 3) Run the container
docker run --rm -p 8501:8501 credit-risk-dashboard
```

Open the app in your browser:
- `http://localhost:8501`

### Python venv Method (local run)

**Prerequisites**
- Python **3.12**
- `pip`

```bash
# 1) Clone the repo and navigate into it
git clone https://github.com/abhi-jith23/Credit-Risk-Dashboard.git
cd Credit-Risk-Dashboard

# 2) Create and activate venv
python3.12 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the Streamlit app
streamlit run app/main_app.py
```

Open:
- `http://localhost:8501`

---

## Reproducibility notes (what was done)

- Dependencies are **pinned** in `requirements.txt`.
- The Docker image uses a fixed base image: `python:3.12.3`.
- Training uses a fixed seed (`random_state=42`) for the train/test split and model settings.
- Data ingestion produces a cached dataset and a lineage metadata file (timestamp + SHA256).
- A data quality report is produced and used as a strict gate before warehouse build and training.
- Models are saved as **versioned artifacts** (no overwriting of prior runs).
- KPIs are defined as **version-controlled SQL** and can be recomputed deterministically from DuckDB tables.

---

## Project Structure

```text
.
├── app
│   ├── main_app.py                      # Streamlit entrypoint (model/version selector + page router)
│   ├── ui_text.py                       # UI labels + category mappings for display
│   └── pages
│       ├── applicant_scoring.py         # Underwriting scoring UI (maps UI columns -> model columns)
│       ├── portfolio_explorer.py        # Portfolio exploration + segmentation (batch scoring)
│       ├── model_card.py                # Governance-style diagnostics (ROC/PR/calibration/importance)
│       ├── data_quality.py              # Data quality report viewer
│       ├── bi_metrics.py                # BI metrics page (reads KPI tables from DuckDB)
│       └── policy_impact.py             # Policy simulator UI (writes policy runs into DuckDB)
├── artifacts
│   ├── ingestion
│   │   └── ingestion_metadata.json      # Lineage metadata (source, timestamp, SHA256, shape)
│   ├── data_quality
│   │   └── data_quality_report.json     # Data contract results (pass/fail + details)
│   ├── models
│   │   ├── logreg/<model_version>/      # Versioned logreg bundles (model + metrics + preds + importance)
│   │   └── lightgbm/<model_version>/
│   └── kpis/<kpi_run_id>/               # KPI export snapshots (CSV + manifest per run)
├── config
│   ├── pipeline.yaml                    # Paths + KPI target selection (model/version/policy_run_id)
│   └── policy.yaml                      # Policy parameters + model selection for policy runs
├── data
│   └── cache/german_credit_raw.csv      # Cached raw dataset used by the pipeline
├── sql
│   ├── ddl/001_create_tables.sql        # Version-controlled DDL for warehouse + KPI tables
│   └── kpis/                            # KPI SQL definitions (funnel, default rate, calibration, segments)
├── src
│   ├── ingest.py                        # Ingestion + lineage metadata
│   ├── validate.py                      # Data contract + fail-closed quality gate
│   ├── warehouse.py                     # DuckDB warehouse build/load
│   ├── train.py                         # Train logreg + lightgbm, save versioned artifacts, write fact_score
│   ├── policy.py                        # Policy simulator (writes policy_run_* tables)
│   ├── kpis.py                          # KPI runner (executes SQL, writes KPI tables, exports CSV snapshots)
│   ├── pipeline.py                      # End-to-end orchestration (no manual copy of IDs)
│   └── models/
│       ├── logreg.py                    # LogReg training module
│       └── lightgbm.py                  # LightGBM training module
├── warehouse/credit_risk.duckdb         # Persistent DuckDB database file
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Data Flow

### 1) Ingestion (retrieve + cache + lineage)

What happens:
- The dataset is downloaded programmatically and cached to `data/cache/german_credit_raw.csv`.
- A lineage file is created at `artifacts/ingestion/ingestion_metadata.json` containing retrieval timestamp and SHA256 hash.

Replicate for your own dataset:
- Replace the cached CSV with your own file under `data/cache/`.
- Update ingestion to write a lineage metadata file with timestamp and SHA256 for your dataset.

Run:
```bash
python -m src.ingest
```

### 2) Validation (data contract + fail-closed gate)

What happens:
- The dataset is validated against a strict schema and constraints.
- A report is written to `artifacts/data_quality/data_quality_report.json`.
- If validation fails, downstream steps are blocked.

Replicate for your own dataset:
- Define your expected schema (columns, types), ranges, and allowed categorical codes.
- Keep the “fail closed” behavior so errors stop the pipeline.

Run:
```bash
python -m src.validate
```

### 3) Warehouse (DuckDB as single source of truth)

What happens:
- A persistent DuckDB database is created at `warehouse/credit_risk.duckdb`.
- Clean tables are created and populated:
  - `dim_applicant`, `fact_outcome`, `fact_score`, `meta_ingestion`

Replicate for your own dataset:
- Create a dimension table for features and a fact table for outcomes using a stable key.

Run:
```bash
python -m src.warehouse
```

### 4) Model training (two models + versioned artifacts)

What happens:
- Two models are trained from DuckDB tables:
  - logreg and lightgbm
- Each training run produces a unique `model_version` and saves a bundle:
  - `model.joblib`, `metrics.json`, `test_predictions.csv`, `feature_importance.csv`
- Scores for the full dataset are written into DuckDB `fact_score` with model identifiers.

Replicate for your own dataset:
- Ensure your warehouse produces the features/labels the training expects.
- Keep the versioned artifact layout and write scores back to the warehouse.

Run:
```bash
python -m src.train
```

### 5) Policy simulation (decisioning + impact)

What happens:
- A policy run reads PD scores from DuckDB for a chosen model/version.
- It applies thresholds and review capacity and computes expected value.
- Outputs are stored in:
  - `policy_run_summary`
  - `policy_run_decisions`

Replicate for your own dataset:
- Use your own PD scores + outcome labels and adjust economics assumptions to your context.

Run:
```bash
python -m src.policy
```

### 6) BI KPIs in SQL (metrics layer)

What happens:
- KPI definitions are stored as SQL files in `sql/kpis/`.
- Running KPIs:
  - materializes KPI tables in DuckDB
  - exports CSV snapshots to `artifacts/kpis/<kpi_run_id>/`

Replicate for your own dataset:
- Replace or extend SQL KPI definitions to match your segments and business metrics.

Run:
```bash
python -m src.kpis
```

### 7) End-to-end pipeline (no manual copy of IDs)

What happens:
- The full workflow is orchestrated in one command:
  - ingest → validate → warehouse → train → policy → kpis
- Configuration is updated automatically to avoid manual copy/paste of versions and run IDs.

Run:
```bash
python -m src.pipeline
```

---

## BI and SQL capabilities

- DuckDB stores all analytics tables and enables BI-style querying.
- KPI definitions are version-controlled in `sql/kpis/` and executed into DuckDB KPI tables:
  - Funnel rates by decision bucket
  - Default rates by decision bucket
  - Calibration deciles (avg PD vs observed default rate)
  - Segment risk (age bucket, housing, purpose)
- KPI outputs are also exported as CSV snapshots per KPI run for reproducibility and sharing.

---

## Key Improvements in the Next Update

Integrated ingestion from user-provided files (CSV or similar) with automatic refresh
- Add an ingestion mode that accepts a dataset file directly (instead of only UCI retrieval).
- Automatically rebuild the warehouse and retrain models when new data is ingested.
- Automatically recompute policy runs and refresh KPI tables and exports after retraining.
- Reduce manual configuration edits by selecting the latest model version and latest policy run for KPI computation.

---

## Disclaimer

This dashboard is an **educational demonstration** of a credit risk workflow.

- It is **not** a production-grade credit scoring system.
- Outputs are based on a historical academic dataset and demonstration models.
- Do not use this tool to make real lending decisions, credit approvals, or compliance judgments.
- Real-world credit risk systems require stronger validation, monitoring, governance, security, and fairness checks.

---

## Author

**Abhijith Senthilkumar**  
*MSc Data Science, University of Luxembourg*

- GitHub: `https://github.com/abhi-jith23`
- Email: `abhijith.unilu@gmail.com`