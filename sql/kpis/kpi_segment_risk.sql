-- sql/kpis/kpi_segment_risk.sql
-- Segment risk using dim_applicant + fact_outcome (Phase 4) and ties it to a model version for tracking

DELETE FROM kpi_segment_risk
WHERE model_name = '{{MODEL_NAME}}' AND model_version = '{{MODEL_VERSION}}';

-- Age bucket
INSERT INTO kpi_segment_risk
SELECT
    '{{MODEL_NAME}}' AS model_name,
    '{{MODEL_VERSION}}' AS model_version,
    'age_bucket' AS segment_type,
    CASE
        WHEN a.age_years < 25 THEN '<25'
        WHEN a.age_years < 35 THEN '25-34'
        WHEN a.age_years < 50 THEN '35-49'
        ELSE '50+'
    END AS segment_value,
    COUNT(*) AS n,
    AVG(CASE WHEN o.credit_risk = 2 THEN 1 ELSE 0 END * 1.0) AS default_rate
FROM dim_applicant a
JOIN fact_outcome o USING(application_id)
GROUP BY segment_value;

-- Housing
INSERT INTO kpi_segment_risk
SELECT
    '{{MODEL_NAME}}' AS model_name,
    '{{MODEL_VERSION}}' AS model_version,
    'housing' AS segment_type,
    a.housing AS segment_value,
    COUNT(*) AS n,
    AVG(CASE WHEN o.credit_risk = 2 THEN 1 ELSE 0 END * 1.0) AS default_rate
FROM dim_applicant a
JOIN fact_outcome o USING(application_id)
GROUP BY a.housing;

-- Purpose
INSERT INTO kpi_segment_risk
SELECT
    '{{MODEL_NAME}}' AS model_name,
    '{{MODEL_VERSION}}' AS model_version,
    'purpose' AS segment_type,
    a.purpose AS segment_value,
    COUNT(*) AS n,
    AVG(CASE WHEN o.credit_risk = 2 THEN 1 ELSE 0 END * 1.0) AS default_rate
FROM dim_applicant a
JOIN fact_outcome o USING(application_id)
GROUP BY a.purpose;
