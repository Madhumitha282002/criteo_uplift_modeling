"""Tests for Qini curve and coefficient computation."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.qini import compute_qini_coefficient, compute_qini_curve


def _make_criteo_like_data(seed: int, n: int = 100_000):
    """Generate a test dataset matching Criteo's regime:
    rare conversions (~0.3%), 85/15 treatment split."""
    rng = np.random.default_rng(seed=seed)
    uplift_pred = rng.uniform(-1, 1, size=n)
    y = (rng.uniform(size=n) < 0.003).astype("int8")  # ~0.3% positive
    w = (rng.uniform(size=n) < 0.85).astype("int8")  # 85% treatment
    return uplift_pred, y, w


def test_random_predictor_gives_near_zero_qini():
    """Random predictor's Qini should be near zero in expectation.

    Tests that the Qini estimator is unbiased: averaging over many
    random-predictor trials should give a mean indistinguishable from
    zero. Individual trial noise (the std) varies with sample size and
    conversion rate; we don't assert on its magnitude, only that the
    mean across trials is consistent with zero.
    """
    n_trials = 20
    qini_values = []
    for seed in range(n_trials):
        uplift_pred, y, w = _make_criteo_like_data(seed=seed)
        qini_values.append(compute_qini_coefficient(uplift_pred, y, w))

    qini_arr = np.array(qini_values)
    mean_q = qini_arr.mean()
    std_q = qini_arr.std()
    # Standard error of the mean = std / sqrt(n_trials)
    se_mean = std_q / np.sqrt(n_trials)

    # If the estimator is unbiased, the mean across trials should be
    # within a few standard errors of zero. Use 4-sigma to keep the test
    # robust to seed-dependent variation (false-positive rate ~0.006%).
    assert abs(mean_q) < 4 * se_mean, (
        f"Mean Qini across {n_trials} random trials = {mean_q:.4f}, "
        f"SE of mean = {se_mean:.4f}. Mean is >4 SE from zero — "
        f"implementation may be biased."
    )


def test_perfect_predictor_beats_random():
    """A predictor with perfect ranking should give Qini >> random."""
    rng = np.random.default_rng(seed=42)
    n = 10_000
    is_responder = np.zeros(n, dtype="int8")
    is_responder[: n // 2] = 1
    rng.shuffle(is_responder)

    w = (rng.uniform(size=n) < 0.5).astype("int8")
    y = (is_responder & w).astype("int8")

    perfect_uplift = is_responder.astype("float64")
    q_perfect = compute_qini_coefficient(perfect_uplift, y, w)

    random_uplift = rng.uniform(-1, 1, size=n)
    q_random = compute_qini_coefficient(random_uplift, y, w)

    assert q_perfect > 10 * abs(q_random), (
        f"Perfect predictor (Q={q_perfect:.4f}) should dominate random "
        f"(Q={q_random:.4f})"
    )
    assert q_perfect > 0, f"Perfect predictor got negative Qini={q_perfect:.4f}"


def test_curve_shapes():
    """Curve and baseline have matching length n_bins+1."""
    uplift_pred, y, w = _make_criteo_like_data(seed=42)
    k_pct, curve, baseline = compute_qini_curve(uplift_pred, y, w, n_bins=100)

    assert len(k_pct) == 101
    assert len(curve) == 101
    assert len(baseline) == 101

    # Curve starts at zero
    assert curve[0] == 0.0
    assert baseline[0] == 0.0

    # k_pct spans [0, 1] inclusive
    assert k_pct[0] == 0.0
    assert k_pct[-1] == 1.0

    # Baseline is linear: equal step sizes
    diffs = np.diff(baseline)
    np.testing.assert_allclose(diffs, diffs[0], atol=1e-12)


def test_input_validation():
    """Bad inputs raise ValueError."""
    n = 100
    uplift = np.zeros(n)
    y = np.zeros(n)
    w = np.zeros(n)

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_qini_coefficient(uplift, y[:50], w)

    with pytest.raises(ValueError, match="w must"):
        compute_qini_coefficient(uplift, y, np.full(n, 2))

    with pytest.raises(ValueError, match="y must"):
        compute_qini_coefficient(uplift, np.full(n, 2), w)
