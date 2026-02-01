-- sql/kpis/kpi_default_rate.sql
-- Default rate by decision bucket (Phase 7)

DELETE FROM kpi_default_rate_by_bucket WHERE policy_run_id = '{{POLICY_RUN_ID}}';

INSERT INTO kpi_default_rate_by_bucket
SELECT
    '{{POLICY_RUN_ID}}' AS policy_run_id,
    decision_bucket,
    COUNT(*) AS n,
    AVG(true_default * 1.0) AS default_rate
FROM policy_run_decisions
WHERE policy_run_id = '{{POLICY_RUN_ID}}'
GROUP BY decision_bucket;
