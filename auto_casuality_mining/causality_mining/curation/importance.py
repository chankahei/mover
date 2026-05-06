"""Per-column feature importance.

Two implementations are exposed; both return a dict mapping column name to a
non-negative score:

- `permutation_importance`: model-agnostic baseline that always works.
- `shap_importance`: optional, used if `shap` is installed and the model is a
  tree ensemble.

The architecture doc names SHAP explicitly for candidate-edge proposal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as _sk_perm


def permutation_importance(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 0,
) -> dict[str, float]:
    """Mean drop in R^2 / accuracy when each column is independently shuffled."""
    result = _sk_perm(model, x, y, n_repeats=n_repeats, random_state=random_state)
    return {col: float(max(0.0, m)) for col, m in zip(x.columns, result.importances_mean)}


def shap_importance(model: Any, x: pd.DataFrame) -> dict[str, float] | None:
    """Mean(|SHAP|) per feature column, or `None` if `shap` is not available."""
    try:
        import shap
    except ImportError:
        return None
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(v) for v in shap_values], axis=0)
    abs_mean = np.mean(np.abs(shap_values), axis=0)
    return {col: float(v) for col, v in zip(x.columns, abs_mean)}
