-- sql/kpis/kpi_funnel.sql
-- Uses policy_run_decisions (Phase 7)

DELETE FROM kpi_funnel WHERE policy_run_id = '{{POLICY_RUN_ID}}';

INSERT INTO kpi_funnel
SELECT
    '{{POLICY_RUN_ID}}' AS policy_run_id,
    decision_bucket,
    COUNT(*) AS n,
    COUNT(*) * 1.0 / (SELECT COUNT(*) FROM policy_run_decisions WHERE policy_run_id = '{{POLICY_RUN_ID}}') AS share
FROM policy_run_decisions
WHERE policy_run_id = '{{POLICY_RUN_ID}}'
GROUP BY decision_bucket;
