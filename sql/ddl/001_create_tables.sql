-- sql/ddl/001_create_tables.sql

-- Core tables (also created by src/warehouse.py, kept here for version-controlled DDL)
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

CREATE TABLE IF NOT EXISTS fact_outcome (
    application_id INTEGER PRIMARY KEY,
    credit_risk INTEGER
);

CREATE TABLE IF NOT EXISTS fact_score (
    application_id INTEGER,
    model_name VARCHAR,
    model_version VARCHAR,
    pd DOUBLE,
    scored_at_utc VARCHAR,
    policy_version VARCHAR,
    decision_bucket VARCHAR
);

CREATE TABLE IF NOT EXISTS meta_ingestion (
    run_id VARCHAR PRIMARY KEY,
    source VARCHAR,
    ingested_at_utc VARCHAR,
    sha256 VARCHAR,
    rows INTEGER,
    cols INTEGER,
    cache_csv VARCHAR
);

-- Policy tables (created by src/policy.py, also version-controlled here)
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

CREATE TABLE IF NOT EXISTS policy_run_decisions (
    policy_run_id VARCHAR,
    application_id INTEGER,
    pd DOUBLE,
    true_default INTEGER,
    decision_bucket VARCHAR,
    review_processed INTEGER,
    expected_value DOUBLE
);

-- KPI tables
CREATE TABLE IF NOT EXISTS kpi_funnel (
    policy_run_id VARCHAR,
    decision_bucket VARCHAR,
    n INTEGER,
    share DOUBLE
);

CREATE TABLE IF NOT EXISTS kpi_default_rate_by_bucket (
    policy_run_id VARCHAR,
    decision_bucket VARCHAR,
    n INTEGER,
    default_rate DOUBLE
);

CREATE TABLE IF NOT EXISTS kpi_calibration_decile (
    model_name VARCHAR,
    model_version VARCHAR,
    decile INTEGER,
    n INTEGER,
    avg_pd DOUBLE,
    observed_default_rate DOUBLE
);

CREATE TABLE IF NOT EXISTS kpi_segment_risk (
    model_name VARCHAR,
    model_version VARCHAR,
    segment_type VARCHAR,
    segment_value VARCHAR,
    n INTEGER,
    default_rate DOUBLE
);
