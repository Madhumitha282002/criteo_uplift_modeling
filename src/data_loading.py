"""Reusable loader for pickled modeling arrays.

Day 8 prepared X, W, y arrays for train and test, plus the fitted
StandardScaler. This module exposes a single function to load them
into a clean dict for use in modeling notebooks and scripts.

Example:
    from src.data_loading import load_processed_data
    data = load_processed_data()
    X_train, y_train = data["X_train"], data["y_train_conv"]
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Resolve relative to this file's location — works regardless of caller's CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


_ARTIFACT_NAMES = (
    "X_train",
    "X_test",
    "W_train",
    "W_test",
    "y_train_conv",
    "y_test_conv",
    "y_train_visit",
    "y_test_visit",
    "row_id_train",
    "row_id_test",
    "scaler",
    "feature_cols",
)


def load_processed_data(processed_dir: Path | None = None) -> dict[str, Any]:
    """Load all pickled modeling artifacts into a single dict.

    Args:
        processed_dir: Directory containing the .pkl files. Defaults to
            ``<project_root>/data/processed``.

    Returns:
        A dict with the following keys:
            X_train, X_test:           float32 arrays of shape (n, 12)
            W_train, W_test:           int8 arrays of shape (n,) — treatment
            y_train_conv, y_test_conv: int8 arrays — conversion outcome
            y_train_visit, y_test_visit: int8 arrays — visit outcome
            row_id_train, row_id_test: int64 arrays — for SQL joins
            scaler:                    fitted StandardScaler (fit on train only)
            feature_cols:              list of 12 feature name strings

    Raises:
        FileNotFoundError: if any expected pickle is missing. Run
            ``notebooks/03_modeling.ipynb`` Day 8 cells to (re)generate.
    """
    base = Path(processed_dir) if processed_dir else _PROCESSED_DIR
    if not base.exists():
        raise FileNotFoundError(
            f"Processed-data directory not found: {base}. "
            "Run notebooks/03_modeling.ipynb Day 8 cells first."
        )

    out: dict[str, Any] = {}
    for name in _ARTIFACT_NAMES:
        path = base / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pickle: {path}. "
                "Run notebooks/03_modeling.ipynb Day 8 cells first."
            )
        with open(path, "rb") as fh:
            out[name] = pickle.load(fh)
    return out
