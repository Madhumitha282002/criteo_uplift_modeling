"""Shape and contract tests for the T-learner module.

Day 9: the implementations raise NotImplementedError. The shape test
is marked xfail today; it will pass automatically on Day 10 once the
real implementation lands.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.t_learner import fit_t_learner, predict_uplift, predict_outcomes


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


@pytest.mark.xfail(
    reason="Day 9: skeletons raise NotImplementedError. Will pass Day 10.",
    raises=NotImplementedError,
    strict=True,
)
def test_t_learner_shapes(dummy_data):
    """Output shapes match the documented contract.

    On Day 10 once fit_t_learner is implemented, this test should pass
    without modification. The strict=True ensures the test fails loudly
    if it starts unexpectedly succeeding (which is the signal that we
    can flip xfail off).
    """
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