"""Decile-level analysis of uplift predictions.

Buckets the test set into 10 equal-sized groups by predicted uplift,
then computes per-decile observed lift and feature profile. The
top-vs-bottom-decile contrast is the most intuitive uplift
visualization for non-technical readers.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def compute_decile_profile(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    uplift_col: str,
    feature_cols: list[str],
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Compute per-decile aggregates by predicted-uplift rank.

    Each row in the returned DataFrame summarizes one decile of the test
    set. Decile 1 = highest predicted uplift, decile 10 = lowest.

    Args:
        con: Active DuckDB connection. ``table_name`` must exist and
            contain ``treatment``, ``conversion``, ``visit``, the
            ``uplift_col``, and every column in ``feature_cols``.
        table_name: SQL table with predictions joined to outcomes.
        uplift_col: Column name of predicted uplift (e.g. ``uplift_gbm``).
        feature_cols: Feature columns to average per decile (e.g. ``f0``..``f11``).
        n_deciles: Number of equal-sized rank buckets. Default 10.

    Returns:
        DataFrame with one row per decile and columns:
            decile, n, n_treated, n_control, treatment_rate,
            conversion_rate, visit_rate,
            treated_conv_rate, control_conv_rate,
            observed_lift, observed_lift_se,
            mean_uplift_pred,
            <one column per feature>.
        Sorted by decile ascending (decile 1 first).
    """
    feature_avgs = ",\n        ".join(f"AVG({c}) AS mean_{c}" for c in feature_cols)

    query = f"""
    WITH ranked AS (
        SELECT
            *,
            NTILE({n_deciles}) OVER (ORDER BY {uplift_col} DESC) AS decile
        FROM {table_name}
    ),
    per_decile AS (
        SELECT
            decile,
            COUNT(*)                                           AS n,
            SUM(CAST(treatment AS BIGINT))                     AS n_treated,
            SUM(1 - CAST(treatment AS BIGINT))                 AS n_control,
            AVG(CAST(treatment AS DOUBLE))                     AS treatment_rate,
            AVG(CAST(conversion AS DOUBLE))                    AS conversion_rate,
            AVG(CAST(visit AS DOUBLE))                         AS visit_rate,
            AVG({uplift_col})                                  AS mean_uplift_pred,
            -- Per-arm rates for lift calculation
            SUM(CASE WHEN treatment = 1 THEN conversion ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN treatment = 1 THEN 1 ELSE 0 END), 0)
                AS treated_conv_rate,
            SUM(CASE WHEN treatment = 0 THEN conversion ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN treatment = 0 THEN 1 ELSE 0 END), 0)
                AS control_conv_rate,
            {feature_avgs}
        FROM ranked
        GROUP BY decile
    )
    SELECT
        *,
        treated_conv_rate - control_conv_rate AS observed_lift,
        -- Standard error of difference of two proportions
        SQRT(
            treated_conv_rate * (1 - treated_conv_rate)
                / NULLIF(n_treated, 0)
          + control_conv_rate * (1 - control_conv_rate)
                / NULLIF(n_control, 0)
        ) AS observed_lift_se
    FROM per_decile
    ORDER BY decile
    """
    return con.execute(query).fetchdf()
