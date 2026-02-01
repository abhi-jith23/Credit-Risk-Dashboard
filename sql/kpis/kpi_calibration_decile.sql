-- sql/kpis/kpi_calibration_decile.sql
-- Calibration deciles for a specific model_name/model_version (Phase 6)

DELETE FROM kpi_calibration_decile
WHERE model_name = '{{MODEL_NAME}}' AND model_version = '{{MODEL_VERSION}}';

WITH scored AS (
    SELECT
        s.application_id,
        s.pd,
        CASE WHEN o.credit_risk = 2 THEN 1 ELSE 0 END AS true_default
    FROM fact_score s
    JOIN fact_outcome o USING(application_id)
    WHERE s.model_name = '{{MODEL_NAME}}'
      AND s.model_version = '{{MODEL_VERSION}}'
),
binned AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY pd) AS decile
    FROM scored
)
INSERT INTO kpi_calibration_decile
SELECT
    '{{MODEL_NAME}}' AS model_name,
    '{{MODEL_VERSION}}' AS model_version,
    decile,
    COUNT(*) AS n,
    AVG(pd) AS avg_pd,
    AVG(true_default * 1.0) AS observed_default_rate
FROM binned
GROUP BY decile
ORDER BY decile;
