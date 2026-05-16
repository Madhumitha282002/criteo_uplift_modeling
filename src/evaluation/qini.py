"""Qini curve and coefficient for uplift evaluation.

Implements the Radcliffe & Surry (2011) Qini formulation from scratch:

    incremental(K)  =  Σ_{top K, treated} y
                     − Σ_{top K, control} y × (n_treated_top_K / n_control_top_K)

The reweighting term corrects for unequal treated/control sizes within
the top-K slice. Day 11 methodology writeup explains the derivation;
this module just implements it.
"""

from __future__ import annotations

import numpy as np


def compute_qini_curve(
    uplift_pred: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Qini curve points and random baseline.

    Sorts all rows by ``uplift_pred`` descending, then computes
    cumulative incremental conversions at each of ``n_bins + 1`` cutoffs
    from K=0 to K=1.

    Args:
        uplift_pred: Predicted uplift τ̂(x), shape ``(n,)``.
        y: Binary outcomes, shape ``(n,)``. Values in {0, 1}.
        w: Treatment assignment, shape ``(n,)``. Values in {0, 1}.
        n_bins: Number of cutoff bins. The returned arrays have length
            ``n_bins + 1`` (cutoff 0 through cutoff 1 inclusive).

    Returns:
        Tuple ``(k_pct, qini_curve, random_baseline)``:
            - ``k_pct``: fraction targeted at each cutoff, shape ``(n_bins+1,)``.
              Values in [0, 1].
            - ``qini_curve``: cumulative incremental conversions at each
              cutoff, shape ``(n_bins+1,)``. ``qini_curve[0] = 0``.
            - ``random_baseline``: straight line from (0, 0) to
              (1, qini_curve[-1]), shape ``(n_bins+1,)``.

    Raises:
        ValueError: If lengths are inconsistent or ``w`` / ``y`` contain
            values outside {0, 1}.
    """
    if not (len(uplift_pred) == len(y) == len(w)):
        raise ValueError(
            f"Length mismatch: uplift={len(uplift_pred)}, " f"y={len(y)}, w={len(w)}"
        )
    if not set(np.unique(w).tolist()).issubset({0, 1}):
        raise ValueError("w must contain only {0, 1}")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("y must contain only {0, 1}")

    n = len(uplift_pred)

    # Sort descending by predicted uplift. Use a stable sort so ties are
    # resolved deterministically — important for reproducibility tests.
    order = np.argsort(-uplift_pred, kind="stable")
    y_sorted = y[order].astype("float64")
    w_sorted = w[order].astype("float64")

    # Cumulative sums up to each row position
    cum_y_treated = np.cumsum(y_sorted * w_sorted)
    cum_y_control = np.cumsum(y_sorted * (1 - w_sorted))
    cum_n_treated = np.cumsum(w_sorted)
    cum_n_control = np.cumsum(1 - w_sorted)

    # Choose row positions to evaluate at. Linspace over [0, n] gives
    # n_bins + 1 cutoffs; each cutoff is the number of top rows to include.
    cut_positions = np.linspace(0, n, n_bins + 1).astype(int)

    k_pct = cut_positions / n  # Fraction targeted, in [0, 1]
    qini_curve = np.zeros(n_bins + 1, dtype="float64")

    for i, pos in enumerate(cut_positions):
        if pos == 0:
            qini_curve[i] = 0.0
            continue
        sum_y_t = cum_y_treated[pos - 1]
        sum_y_c = cum_y_control[pos - 1]
        n_t = cum_n_treated[pos - 1]
        n_c = cum_n_control[pos - 1]

        # Reweighting: scale control sum by treated/control ratio.
        # If one arm is empty in the top slice, fall back to 0 reweighting —
        # i.e., we can't estimate the incremental count from a missing arm.
        if n_c == 0:
            # No control rows in slice: report treated conversions alone.
            qini_curve[i] = sum_y_t
        else:
            qini_curve[i] = sum_y_t - sum_y_c * (n_t / n_c)

    # Random baseline: straight line from origin to the final qini value
    random_baseline = k_pct * qini_curve[-1]

    return k_pct, qini_curve, random_baseline


def compute_qini_coefficient(
    uplift_pred: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_bins: int = 100,
) -> float:
    """Compute the Qini coefficient: area between curve and random baseline.

    Positive values indicate the ranking beats random targeting; negative
    values mean the ranking is worse than random. The magnitude depends on
    the dataset's underlying heterogeneity — on Criteo, well-performing
    models typically score in the 0.005–0.015 range.

    Args:
        uplift_pred: Predicted uplift τ̂(x), shape ``(n,)``.
        y: Binary outcomes, shape ``(n,)``.
        w: Treatment assignment, shape ``(n,)``.
        n_bins: Discretization for the trapezoidal integral. 100 is
            sufficient for stable estimates on test sets >50k rows.

    Returns:
        Qini coefficient as a float. Computed via the trapezoidal
        rule over (k_pct, qini_curve − random_baseline).
    """
    k_pct, qini_curve, random_baseline = compute_qini_curve(
        uplift_pred, y, w, n_bins=n_bins
    )
    diff = qini_curve - random_baseline
    # np.trapz is deprecated in numpy 2.0+ in favor of np.trapezoid;
    # try the new name first and fall back for older numpy.
    return float(np.trapezoid(diff, k_pct))
