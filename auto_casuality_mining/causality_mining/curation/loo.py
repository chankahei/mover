"""Per-target leave-one-out fit with LightGBM + Tree SHAP.

For one target column, fit ONE model PER LAG. Each model predicts the
forward-`lag`-step change of the target at time t from the (already backward
1-step changed) feature columns at time t. SHAP gives per-feature importance
+ signed strength for that lag.

The caller assembles per-feature scores from each lag into edge candidates;
this module only owns the fit + SHAP extraction for a single (target, lag).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from causality_mining.timeseries.kind import TimeSeriesKind


@dataclass(frozen=True)
class FeatureScore:
    """Score of one feature column for one target at one lag."""

    importance: float
    signed_strength: float


_TARGET_COL = "__target__"


def fit_one_lag(
    feature_panel: pd.DataFrame,
    target_series: pd.Series,
    target_kind: TimeSeriesKind,
    feature_cols: Sequence[str],
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    random_state: int = 0,
) -> tuple[dict[str, FeatureScore], int]:
    """Fit a LightGBM model for one (target, lag) pair and return SHAP scores.

    `target_series` must already be the forward-change of the original target
    at the lag of interest, indexed on the same grid as `feature_panel`.
    """
    if not feature_cols:
        return {}, 0

    df = pd.concat(
        [feature_panel[list(feature_cols)], target_series.rename(_TARGET_COL)],
        axis=1,
    ).dropna()
    if df.empty or df[_TARGET_COL].nunique() < 2:
        return {}, 0

    x = df.drop(columns=[_TARGET_COL]).astype(float)
    y_raw = df[_TARGET_COL]

    common = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        verbose=-1,
    )
    model = _fit_lgbm(x.values, y_raw, target_kind, common)
    if model is None:
        return {}, len(df)

    shap_matrix = _tree_shap_matrix(model, x.values)
    if shap_matrix is None:
        return _gain_scores(model, x.columns, n_rows=len(df)), len(df)

    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    mean_signed = np.mean(shap_matrix, axis=0)
    scores = {
        col: FeatureScore(importance=float(mean_abs[i]), signed_strength=float(mean_signed[i]))
        for i, col in enumerate(x.columns)
    }
    return scores, len(df)


def _gain_scores(model, columns, n_rows: int) -> dict[str, FeatureScore]:
    """Fall back to model-reported gain importance when SHAP isn't available."""
    gains = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    if gains.size != len(columns) or gains.sum() <= 0:
        return {}
    normalized = gains / gains.sum()
    return {
        col: FeatureScore(importance=float(normalized[i]), signed_strength=0.0)
        for i, col in enumerate(columns)
    }


def _fit_lgbm(
    x: np.ndarray,
    y_raw: pd.Series,
    target_kind: TimeSeriesKind,
    common: dict,
):
    """Fit either an LGBMRegressor or LGBMClassifier; None on degenerate label."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError:
        return None
    if target_kind is TimeSeriesKind.CATEGORICAL:
        codes, _ = pd.factorize(y_raw)
        if len(set(codes)) < 2:
            return None
        objective = "binary" if len(set(codes)) == 2 else "multiclass"
        model = LGBMClassifier(objective=objective, **common)
        model.fit(x, codes)
        return model
    y = np.asarray(y_raw, dtype=float)
    model = LGBMRegressor(objective="regression", **common)
    model.fit(x, y)
    return model


def _tree_shap_matrix(model, x: np.ndarray) -> np.ndarray | None:
    """Return a `(n_samples, n_features)` SHAP matrix using `TreeExplainer`."""
    try:
        import shap
    except ImportError:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x)
    except Exception:
        return None
    arr = _flatten_shap(values)
    if arr is None or arr.ndim != 2 or arr.shape != (x.shape[0], x.shape[1]):
        return None
    return arr


def _flatten_shap(values) -> np.ndarray | None:
    """Reduce a multiclass shap output to a single (n_samples, n_features) matrix."""
    if isinstance(values, list):
        if not values:
            return None
        if len(values) == 2:
            return np.asarray(values[1])
        return np.mean(np.stack([np.asarray(v) for v in values], axis=0), axis=0)
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            return arr[:, :, 1]
        return np.mean(arr, axis=2)
    return arr
