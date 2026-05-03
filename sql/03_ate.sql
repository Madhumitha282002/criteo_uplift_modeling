-- ATE for visit and conversion outcomes.
-- Computed as difference of two proportions with analytic SE.
-- Run from project root: duckdb data/criteo.duckdb < sql/03_ate.sql

WITH outcome_stats AS (
    SELECT
        outcome,
        SUM(CASE WHEN treatment = 1 THEN y ELSE 0 END) * 1.0
            / SUM(CASE WHEN treatment = 1 THEN 1 ELSE 0 END) AS p_t,
        SUM(CASE WHEN treatment = 0 THEN y ELSE 0 END) * 1.0
            / SUM(CASE WHEN treatment = 0 THEN 1 ELSE 0 END) AS p_c,
        SUM(CASE WHEN treatment = 1 THEN 1 ELSE 0 END)        AS n_t,
        SUM(CASE WHEN treatment = 0 THEN 1 ELSE 0 END)        AS n_c
    FROM (
        SELECT 'visit' AS outcome,      treatment, visit       AS y FROM criteo
        UNION ALL
        SELECT 'conversion' AS outcome, treatment, conversion  AS y FROM criteo
    ) t
    GROUP BY outcome
)
SELECT
    outcome,
    p_c                                                AS control_rate,
    p_t                                                AS treatment_rate,
    p_t - p_c                                          AS ate,
    SQRT( p_t * (1 - p_t) / n_t
        + p_c * (1 - p_c) / n_c )                      AS se,
    (p_t - p_c) - 1.96 * SQRT( p_t * (1 - p_t) / n_t
                              + p_c * (1 - p_c) / n_c) AS ci_low,
    (p_t - p_c) + 1.96 * SQRT( p_t * (1 - p_t) / n_t
                              + p_c * (1 - p_c) / n_c) AS ci_high,
    n_t,
    n_c
FROM outcome_stats
ORDER BY outcome;
