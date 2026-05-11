"""Top-K uplift evaluation for business comparisons.

The Qini coefficient (Day 12) is an integral over all K; uplift-at-top-K
returns the incremental conversions at one specific cutoff. Same
underlying formula:

    incremental(K) = Σ_{top K, treated} y
                   − Σ_{top K, control} y × (n_treated_top_K / n_control_top_K)

Day 13's three-strategy table uses this to compare treat-everyone vs
random-K vs model-K targeting on incremental-conversions-per-cost basis.
"""

from __future__ import annotations

import numpy as np


def uplift_at_top_k(
    uplift_pred: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    k_pct: float,
) -> float:
    """Compute incremental conversions in the top-K%-by-predicted-uplift slice.

    Args:
        uplift_pred: Predicted uplift τ̂(x), shape ``(n,)``.
        y: Binary outcomes, shape ``(n,)``. Values in {0, 1}.
        w: Treatment assignment, shape ``(n,)``. Values in {0, 1}.
        k_pct: Cutoff fraction in (0, 1]. ``k_pct=0.20`` selects the
            top 20% of rows by predicted uplift.

    Returns:
        Incremental conversions in the top-K slice, computed via the
        Radcliffe-Surry reweighting formula. May be negative if the
        model's top-K is worse than random within that slice.

    Raises:
        ValueError: For input length mismatches, invalid labels/treatment,
            or ``k_pct`` outside (0, 1].
    """
    if not (len(uplift_pred) == len(y) == len(w)):
        raise ValueError(
            f"Length mismatch: uplift={len(uplift_pred)}, "
            f"y={len(y)}, w={len(w)}"
        )
    if not (0 < k_pct <= 1):
        raise ValueError(f"k_pct must be in (0, 1], got {k_pct}")
    if not set(np.unique(w).tolist()).issubset({0, 1}):
        raise ValueError("w must contain only {0, 1}")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("y must contain only {0, 1}")

    n = len(uplift_pred)
    k = max(1, int(round(n * k_pct)))

    order = np.argsort(-uplift_pred, kind="stable")
    top_idx = order[:k]

    y_top = y[top_idx].astype("float64")
    w_top = w[top_idx].astype("float64")

    sum_y_t = (y_top * w_top).sum()
    sum_y_c = (y_top * (1 - w_top)).sum()
    n_t = w_top.sum()
    n_c = (1 - w_top).sum()

    if n_c == 0:
        # No control in slice: can't estimate counterfactual. Return treated count.
        return float(sum_y_t)
    return float(sum_y_t - sum_y_c * (n_t / n_c))


def strategy_comparison(
    uplift_pred: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    k_pct: float = 0.20,
) -> dict:
    """Compare three targeting strategies at a single budget level.

    Strategies compared:
        1. **Treat everyone** (K=100%): no targeting, treat the whole
           population. Cost = n_test units.
        2. **Random targeting** (K=k_pct): treat a uniformly random
           fraction. Cost = n_test * k_pct units.
        3. **Model targeting** (K=k_pct): treat the top fraction by
           predicted uplift. Same cost as random.

    Args:
        uplift_pred: Predicted uplift τ̂(x), shape ``(n,)``.
        y: Binary outcomes, shape ``(n,)``.
        w: Treatment assignment, shape ``(n,)``.
        k_pct: Targeted fraction for strategies 2 and 3, in (0, 1].
            Defaults to 0.20.

    Returns:
        Dict with one entry per strategy. Each entry has keys:
            - ``cost``: number of users treated.
            - ``incremental``: estimated incremental conversions.
            - ``incr_per_cost``: incremental / cost (the ROI metric).
            - ``k_pct``: fraction targeted.
    """
    n = len(uplift_pred)
    total_incr_at_100 = uplift_at_top_k(uplift_pred, y, w, k_pct=1.0)
    model_incr_at_k = uplift_at_top_k(uplift_pred, y, w, k_pct=k_pct)

    cost_full = float(n)
    cost_k = float(n * k_pct)
    # Random targeting at k_pct produces k_pct of the total incremental in expectation
    random_incr_at_k = total_incr_at_100 * k_pct

    return {
        "treat_everyone": {
            "k_pct": 1.0,
            "cost": cost_full,
            "incremental": total_incr_at_100,
            "incr_per_cost": total_incr_at_100 / cost_full if cost_full else 0.0,
        },
        f"random_top_{int(k_pct*100)}pct": {
            "k_pct": k_pct,
            "cost": cost_k,
            "incremental": random_incr_at_k,
            "incr_per_cost": random_incr_at_k / cost_k if cost_k else 0.0,
        },
        f"model_top_{int(k_pct*100)}pct": {
            "k_pct": k_pct,
            "cost": cost_k,
            "incremental": model_incr_at_k,
            "incr_per_cost": model_incr_at_k / cost_k if cost_k else 0.0,
        },
    }