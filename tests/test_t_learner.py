"""Shape and contract tests for the T-learner module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.t_learner import (
    fit_t_learner,
    predict_outcomes,
    predict_uplift,
)


@pytest.fixture
def dummy_data():
    """100-row toy dataset: 5 features, ~50/50 treatment, ~10% conversion."""
    rng = np.random.default_rng(seed=42)
    n, d = 100, 5
    X = rng.standard_normal(size=(n, d)).astype("float32")
    w = rng.integers(0, 2, size=n).astype("int8")
    # Inject a small treatment effect so neither arm is degenerate
    base_logit = X[:, 0] * 0.5
    treated_logit = base_logit + 0.3 * w
    p = 1 / (1 + np.exp(-treated_logit))
    y = (rng.uniform(size=n) < p).astype("int8")
    return X, y, w


def test_t_learner_shapes(dummy_data):
    """Output shapes and contracts match the documented behavior."""
    X, y, w = dummy_data

    m_t, m_c = fit_t_learner(
        X, y, w,
        base_model_class=LogisticRegression,
        max_iter=200,
        random_state=42,
    )

    uplift = predict_uplift(m_t, m_c, X)
    p_t, p_c = predict_outcomes(m_t, m_c, X)

    # Shape contract
    assert uplift.shape == (100,), f"uplift wrong shape: {uplift.shape}"
    assert p_t.shape == (100,), f"p_t wrong shape: {p_t.shape}"
    assert p_c.shape == (100,), f"p_c wrong shape: {p_c.shape}"

    # Range contract: probabilities in [0, 1], uplift in [-1, 1]
    assert np.all((p_t >= 0) & (p_t <= 1))
    assert np.all((p_c >= 0) & (p_c <= 1))
    assert np.all((uplift >= -1) & (uplift <= 1))

    # Identity contract: uplift == p_t - p_c (within float tolerance)
    np.testing.assert_allclose(uplift, p_t - p_c, atol=1e-12)


def test_t_learner_validates_inputs():
    """Bad inputs raise ValueError, not silent garbage."""
    X = np.zeros((10, 3))
    y = np.zeros(10)
    w = np.zeros(10)

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        fit_t_learner(X, y[:5], w, LogisticRegression)

    # Bad treatment values (not in {0, 1})
    with pytest.raises(ValueError, match=r"w must be"):
        fit_t_learner(X, y, np.full(10, 2), LogisticRegression)

    # Empty arm: all w=0 leaves the treated arm empty
    with pytest.raises(ValueError, match="Both arms"):
        fit_t_learner(X, y, np.zeros(10), LogisticRegression)


def test_t_learner_with_seeded_models_is_deterministic(dummy_data):
    """Same data + same seed → identical predictions."""
    X, y, w = dummy_data

    m_t1, m_c1 = fit_t_learner(
        X, y, w, base_model_class=LogisticRegression,
        max_iter=200, random_state=42,
    )
    m_t2, m_c2 = fit_t_learner(
        X, y, w, base_model_class=LogisticRegression,
        max_iter=200, random_state=42,
    )

    u1 = predict_uplift(m_t1, m_c1, X)
    u2 = predict_uplift(m_t2, m_c2, X)
    np.testing.assert_allclose(u1, u2, atol=1e-12)