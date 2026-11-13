-- Metric layer. Every dashboard number and every statistical test reads from these views.
-- Release-quality thresholds live in one place (v_thresholds) so they are easy to change.

CREATE OR REPLACE VIEW v_thresholds AS
SELECT 0.925::numeric AS min_pass_rate,   -- pass rate must be at least 92.5%
       4::int          AS max_critical;    -- and at most 4 critical defects found

-- Pass rate = Passed / (Passed + Failed); Blocked runs are reported separately.
CREATE OR REPLACE VIEW v_build_exec AS
SELECT build_id,
       COUNT(*)                                              AS executions,
       COUNT(*) FILTER (WHERE status = 'Passed')             AS passed,
       COUNT(*) FILTER (WHERE status = 'Failed')             AS failed,
       COUNT(*) FILTER (WHERE status = 'Blocked')            AS blocked,
       SUM(duration_min) / 60.0                              AS tester_hours
FROM test_executions
GROUP BY build_id;

CREATE OR REPLACE VIEW v_build_defects AS
SELECT build_id,
       COUNT(*)                                              AS defects,
       COUNT(*) FILTER (WHERE severity = 'Critical')         AS critical_defects,
       COUNT(*) FILTER (WHERE status = 'Open')               AS open_defects,
       COUNT(*) FILTER (WHERE reopen_count > 0)              AS reopened_defects,
       COUNT(*) FILTER (WHERE resolved_at IS NOT NULL)       AS resolved_defects,
       AVG(EXTRACT(EPOCH FROM resolved_at - opened_at) / 86400.0) AS avg_resolution_days
FROM defects
GROUP BY build_id;

CREATE OR REPLACE VIEW v_build_metrics AS
SELECT b.build_id, b.version, b.build_type, b.change_size,
       b.planned_release_date, b.release_date, b.release_delay_days,
       e.executions, e.passed, e.failed, e.blocked,
       ROUND(e.passed::numeric / NULLIF(e.passed + e.failed, 0), 4)           AS pass_rate,
       COALESCE(d.defects, 0)                                                  AS defects,
       ROUND(100.0 * COALESCE(d.defects, 0) / e.executions, 3)                 AS defects_per_100_exec,
       COALESCE(d.critical_defects, 0)                                         AS critical_defects,
       COALESCE(d.open_defects, 0)                                             AS open_defects,
       ROUND(COALESCE(d.reopened_defects, 0)::numeric / NULLIF(d.resolved_defects, 0), 4) AS reopen_rate,
       ROUND(d.avg_resolution_days::numeric, 2)                                AS avg_resolution_days,
       ROUND(e.executions / NULLIF(e.tester_hours, 0)::numeric, 3)             AS executions_per_tester_hour,
       ROUND(COALESCE(d.defects, 0) / NULLIF(e.tester_hours, 0)::numeric, 4)   AS defects_per_tester_hour,
       (e.passed::numeric / NULLIF(e.passed + e.failed, 0) >= t.min_pass_rate
        AND COALESCE(d.critical_defects, 0) <= t.max_critical)                 AS meets_release_quality
FROM builds b
JOIN v_build_exec e USING (build_id)
LEFT JOIN v_build_defects d USING (build_id)
CROSS JOIN v_thresholds t;

-- Build-over-build trend: delta vs previous build and a 5-build rolling pass rate.
CREATE OR REPLACE VIEW v_build_trend AS
SELECT m.*,
       pass_rate - LAG(pass_rate) OVER w                                       AS pass_rate_delta,
       ROUND(AVG(pass_rate) OVER (ORDER BY planned_release_date
                                  ROWS BETWEEN 4 PRECEDING AND CURRENT ROW), 4) AS pass_rate_roll5,
       ROUND(AVG(defects_per_100_exec) OVER (ORDER BY planned_release_date
                                  ROWS BETWEEN 4 PRECEDING AND CURRENT ROW), 3) AS defect_rate_roll5
FROM v_build_metrics m
WINDOW w AS (ORDER BY planned_release_date);

-- Platform view: defect incidence per 100 executions, severity mix.
CREATE OR REPLACE VIEW v_platform_metrics AS
WITH ex AS (
    SELECT platform_id, COUNT(*) AS executions,
           COUNT(*) FILTER (WHERE status = 'Passed') AS passed,
           COUNT(*) FILTER (WHERE status = 'Failed') AS failed
    FROM test_executions GROUP BY platform_id),
df AS (
    SELECT platform_id, COUNT(*) AS defects,
           COUNT(*) FILTER (WHERE severity = 'Critical') AS critical_defects,
           COUNT(*) FILTER (WHERE status = 'Open') AS open_defects
    FROM defects GROUP BY platform_id)
SELECT p.platform_name, ex.executions, ex.passed, ex.failed,
       ROUND(ex.passed::numeric / (ex.passed + ex.failed), 4) AS pass_rate,
       df.defects, df.critical_defects, df.open_defects,
       ROUND(100.0 * df.defects / ex.executions, 3) AS defects_per_100_exec
FROM platforms p JOIN ex USING (platform_id) JOIN df USING (platform_id);

-- Category view: volume, speed, reopen rate and total "blocking days" (a delay proxy).
CREATE OR REPLACE VIEW v_category_metrics AS
SELECT category,
       COUNT(*)                                                                 AS defects,
       COUNT(*) FILTER (WHERE severity = 'Critical')                            AS critical_defects,
       ROUND(AVG(EXTRACT(EPOCH FROM resolved_at - opened_at) / 86400.0)::numeric, 2) AS avg_resolution_days,
       ROUND(COUNT(*) FILTER (WHERE reopen_count > 0)::numeric
             / NULLIF(COUNT(*) FILTER (WHERE resolved_at IS NOT NULL), 0), 4)   AS reopen_rate,
       ROUND(SUM(EXTRACT(EPOCH FROM resolved_at - opened_at) / 86400.0)
             FILTER (WHERE blocks_release)::numeric, 1)                         AS blocking_days
FROM defects
GROUP BY category;

-- Tester efficiency: throughput and yield by team and experience band.
CREATE OR REPLACE VIEW v_tester_efficiency AS
SELECT t.team,
       CASE WHEN t.experience_years < 2 THEN '<2y'
            WHEN t.experience_years < 5 THEN '2-5y' ELSE '5y+' END AS experience_band,
       COUNT(*)                                                     AS executions,
       ROUND(AVG(e.duration_min), 2)                                AS avg_duration_min,
       ROUND(COUNT(*) / (SUM(e.duration_min) / 60.0), 3)            AS executions_per_hour,
       ROUND(100.0 * COUNT(*) FILTER (WHERE e.status = 'Failed') / COUNT(*), 3) AS fail_pct
FROM test_executions e JOIN testers t USING (tester_id)
GROUP BY 1, 2;

-- Execution-level fact table used by the statistics scripts (one row per execution).
CREATE OR REPLACE VIEW v_execution_facts AS
SELECT e.execution_id, e.build_id, e.executed_at, e.platform_id, p.platform_name,
       tc.area, e.status, e.duration_min, t.experience_years, t.team,
       EXISTS (SELECT 1 FROM defects d WHERE d.execution_id = e.execution_id) AS found_defect
FROM test_executions e
JOIN platforms p USING (platform_id)
JOIN test_cases tc USING (test_case_id)
JOIN testers t USING (tester_id);
