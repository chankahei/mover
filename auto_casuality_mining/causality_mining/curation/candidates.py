"""Propose candidate causal edges with lagged predictive models.

For each target, a gradient-boosted model is fit on lagged copies of every
*non-target* feature column. The features whose importance crosses a threshold
become candidate edges. They are *not* causal edges yet -- the refutation step
in `refute.py` decides which to promote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from causality_mining.curation.importance import permutation_importance, shap_importance
from causality_mining.curation.targets import TargetSpec
from causality_mining.panel.builder import ColumnLayout
from causality_mining.panel.lag import lagged_columns


@dataclass(frozen=True)
class CandidateEdge:
    """An edge proposal, carrying enough info to be passed to refutation."""

    source_series: str
    target_series: str
    lag: int
    importance: float
    feature_column: str  # the lagged column actually used by the predictor


def _make_model(target: TargetSpec) -> Any:
    if target.kind == "classification":
        return GradientBoostingClassifier(random_state=0)
    return GradientBoostingRegressor(random_state=0)


def propose_candidates(
    panel: pd.DataFrame,
    layout: ColumnLayout,
    target: TargetSpec,
    target_columns: list[str],
    importance_threshold: float,
    use_shap: bool = True,
) -> tuple[list[CandidateEdge], pd.DataFrame, list[tuple[str, float]]]:
    """Fit a predictor for `target` and return importance-passing candidates.

    Returns:
        (candidates, design_df) where `design_df` contains the lagged feature
        columns and the target column used to fit the predictor. Refutation
        reuses this DataFrame so DoWhy gets the exact same alignment.
    """
    target_col = target_columns[0]
    feature_cols = [c for c in panel.columns if layout.column_to_series[c] != target.series_id]
    lags = list(range(target.min_lag, target.max_lag + 1))
    x_lagged = lagged_columns(panel, feature_cols, lags)
    y = panel[target_col] if len(target_columns) == 1 else panel[target_columns].mean(axis=1)
    design = pd.concat([x_lagged, y.rename(target_col)], axis=1).dropna()
    if design.empty:
        return [], design, []
    x = design.drop(columns=[target_col])
    y_aligned = design[target_col]

    model = _make_model(target)
    model.fit(x, y_aligned)

    importances: dict[str, float] | None = None
    if use_shap:
        importances = shap_importance(model, x)
    if importances is None:
        importances = permutation_importance(model, x, y_aligned)

    ranked_importances = sorted(
        ((col, float(score)) for col, score in importances.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    out: list[CandidateEdge] = []
    for col, score in ranked_importances:
        if score < importance_threshold:
            continue
        base, lag_part = col.rsplit("__lag", 1)
        source = layout.column_to_series[base]
        if source == target.series_id:
            continue
        out.append(
            CandidateEdge(
                source_series=source,
                target_series=target.series_id,
                lag=int(lag_part),
                importance=float(score),
                feature_column=col,
            )
        )
    return out, design, ranked_importances
