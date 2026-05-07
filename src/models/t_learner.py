"""T-learner for uplift modeling.

Implements the two-model approach: train one outcome classifier per
treatment arm, then difference their predictions to estimate the
conditional average treatment effect (CATE):

    τ̂(x) = μ̂_t(x) − μ̂_c(x)

The base model is pluggable — any sklearn-compatible classifier with
``fit`` and ``predict_proba`` will work. Day 10 uses LogisticRegression;
Day 14 swaps in LightGBM with no changes to this module.

Skeletons only on Day 9. Implementation arrives on Day 10.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_t_learner(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    base_model_class: type,
    **model_kwargs: Any,
) -> tuple[Any, Any]:
    """Fit a T-learner: one outcome model per treatment arm.

    Splits ``(X, y)`` by treatment assignment ``w``, then fits a fresh
    instance of ``base_model_class`` to each subset. The two fitted
    models can be differenced via :func:`predict_uplift` to estimate
    treatment effects on new data.

    Args:
        X: Feature matrix of shape ``(n, d)``. Should already be
            scaled if the base model is scale-sensitive.
        y: Binary outcome labels of shape ``(n,)``. Values in {0, 1}.
        w: Binary treatment assignment of shape ``(n,)``.
            1 = treated, 0 = control.
        base_model_class: An sklearn-compatible classifier class
            (not an instance) such as ``LogisticRegression`` or
            ``LGBMClassifier``. Must implement ``fit`` and
            ``predict_proba``.
        **model_kwargs: Hyperparameters forwarded to
            ``base_model_class()`` for both arm models. The same
            kwargs are used for both — pass kwargs that make sense
            given each arm sees a different sample size.

    Returns:
        A tuple ``(model_t, model_c)`` of two fitted classifiers.
        ``model_t`` was fit on rows where ``w == 1``;
        ``model_c`` was fit on rows where ``w == 0``.

    Raises:
        ValueError: If ``X``, ``y``, ``w`` have inconsistent lengths,
            if ``w`` contains values other than {0, 1}, or if either
            treatment arm has zero rows.
    """
    raise NotImplementedError("Implementation arrives Day 10.")


def predict_uplift(
    model_t: Any,
    model_c: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Predict per-individual uplift τ̂(x) = P̂(Y=1|X, W=1) − P̂(Y=1|X, W=0).

    Both models predict the probability of the positive outcome class,
    then we difference. The resulting array can be passed directly to
    ``compute_qini_curve`` and ``uplift_at_top_k`` (Days 12–13).

    Args:
        model_t: Fitted treated-arm model from :func:`fit_t_learner`.
        model_c: Fitted control-arm model from :func:`fit_t_learner`.
        X: Feature matrix of shape ``(n, d)``. Must be scaled the same
            way as training data — apply the saved ``scaler`` from
            Day 8.

    Returns:
        Uplift predictions of shape ``(n,)``, dtype ``float64``.
        Values can be negative — a negative τ̂ means treatment hurts
        that user's predicted outcome rate.
    """
    raise NotImplementedError("Implementation arrives Day 10.")


def predict_outcomes(
    model_t: Any,
    model_c: Any,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the two conditional outcome probabilities separately.

    Useful for diagnostics: comparing ``p_treated`` and ``p_control``
    individually exposes whether one arm's model is degenerate (e.g.
    predicting near-constant values), which would silently break
    uplift estimates if only τ̂ were inspected.

    Args:
        model_t: Fitted treated-arm model from :func:`fit_t_learner`.
        model_c: Fitted control-arm model from :func:`fit_t_learner`.
        X: Feature matrix of shape ``(n, d)``. Must be scaled the same
            way as training data.

    Returns:
        A tuple ``(p_treated, p_control)`` of arrays each shape
        ``(n,)`` and dtype ``float64``. ``p_treated[i]`` is the
        treated-arm model's prediction P̂(Y=1|x_i, W=1);
        ``p_control[i]`` is the control-arm equivalent.
    """
    raise NotImplementedError("Implementation arrives Day 10.")