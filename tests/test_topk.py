"""Tests for top-K uplift functions."""

from __future__ import annotations
import numpy as np
import pytest

from src.evaluation.topk import strategy_comparison, uplift_at_top_k


def _make_data(seed=42, n=10_000):
    rng = np.random.default_rng(seed)
    uplift_pred = rng.uniform(-1, 1, size=n)
    y = (rng.uniform(size=n) < 0.01).astype("int8")
    w = (rng.uniform(size=n) < 0.85).astype("int8")
    return uplift_pred, y, w


def test_top_k_validates_inputs():
    """Bad inputs raise ValueError."""
    uplift, y, w = _make_data()

    with pytest.raises(ValueError, match="k_pct must be"):
        uplift_at_top_k(uplift, y, w, k_pct=0.0)
    with pytest.raises(ValueError, match="k_pct must be"):
        uplift_at_top_k(uplift, y, w, k_pct=1.5)
    with pytest.raises(ValueError, match="Length mismatch"):
        uplift_at_top_k(uplift, y[:5], w, k_pct=0.2)


def test_top_k_at_100pct_equals_total_incremental():
    """At k_pct=1.0, uplift_at_top_k is the unconditional incremental count."""
    uplift, y, w = _make_data()
    incr_100 = uplift_at_top_k(uplift, y, w, k_pct=1.0)

    # Direct computation: same formula on the whole set
    n_t = w.sum()
    n_c = (1 - w).sum()
    sum_y_t = (y * w).sum()
    sum_y_c = (y * (1 - w)).sum()
    expected = sum_y_t - sum_y_c * (n_t / n_c)

    np.testing.assert_allclose(incr_100, expected, atol=1e-9)


def test_strategy_comparison_has_three_strategies():
    """Comparison dict has the three documented strategies."""
    uplift, y, w = _make_data()
    result = strategy_comparison(uplift, y, w, k_pct=0.20)

    assert len(result) == 3
    assert "treat_everyone" in result
    assert "random_top_20pct" in result
    assert "model_top_20pct" in result

    # Each entry has the four documented keys
    for strategy in result.values():
        assert set(strategy.keys()) == {"k_pct", "cost", "incremental", "incr_per_cost"}

    # Random and model at same k have the same cost
    assert result["random_top_20pct"]["cost"] == result["model_top_20pct"]["cost"]
