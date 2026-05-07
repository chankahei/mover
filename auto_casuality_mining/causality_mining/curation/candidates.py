"""Propose candidate causal edges with per-lag predictive models.

For each (target, lag), fit a gradient-boosted model that predicts the forward
`lag`-step change of the target at time t from the (already backward-changed)
feature columns at time t. SHAP/permutation importance picks per-source winners
per lag; the best-importance lag per source becomes the candidate edge.

These are *not* causal edges yet -- the refutation step in `refute.py` decides
which to promote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from causality_mining.curation.forward_target import forward_target_column
from causality_mining.curation.importance import permutation_importance, shap_importance
from causality_mining.curation.targets import TargetSpec
from causality_mining.normalize.base import Change
from causality_mining.panel.builder import ColumnLayout
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class CandidateEdge:
    """An edge proposal, carrying enough info to be passed to refutation."""

    source_series: str
    target_series: str
    lag: int
    importance: float
    feature_column: str  # the panel column actually used by the predictor


def _make_model(target: TargetSpec) -> Any:
    if target.kind == "classification":
        return GradientBoostingClassifier(random_state=0)
    return GradientBoostingRegressor(random_state=0)


def propose_candidates(
    panel: pd.DataFrame,
    layout: ColumnLayout,
    target: TargetSpec,
    raw_target_ts: TimeSeries,
    change: Change,
    freq: str,
    importance_threshold: float,
    use_shap: bool = True,
) -> tuple[list[CandidateEdge], pd.DataFrame, list[tuple[str, float]]]:
    """Fit one predictor per lag in `[target.min_lag, target.max_lag]`.

    Returns:
        (candidates, last_design_df, ranked_importances) where:
          - `candidates` are the per-source winners across lags after threshold.
          - `last_design_df` is the design matrix used for the LAST fitted lag
            (refutation reuses the per-edge design built on demand).
          - `ranked_importances` lists every (panel_column, lag) considered,
            sorted by importance.
    """
    feature_cols = [
        c for c in panel.columns if layout.column_to_series[c] != target.series_id
    ]
    if not feature_cols:
        return [], pd.DataFrame(), []

    lags = list(range(target.min_lag, target.max_lag + 1))
    per_lag_imp: dict[int, dict[str, float]] = {}
    last_design = pd.DataFrame()
    for lag in lags:
        target_series = forward_target_column(raw_target_ts, change, freq, lag)
        design = pd.concat(
            [panel[feature_cols], target_series.rename(target.series_id)],
            axis=1,
        ).dropna()
        if design.empty or design[target.series_id].nunique() < 2:
            continue
        x = design.drop(columns=[target.series_id])
        y = design[target.series_id]

        model = _make_model(target)
        model.fit(x, y)

        imps = shap_importance(model, x) if use_shap else None
        if imps is None:
            imps = permutation_importance(model, x, y)
        per_lag_imp[lag] = imps
        last_design = design

    ranked: list[tuple[str, int, float]] = []
    for lag, imps in per_lag_imp.items():
        for col, score in imps.items():
            ranked.append((col, lag, float(score)))
    ranked.sort(key=lambda item: item[2], reverse=True)

    best_per_source: dict[str, tuple[int, float, str]] = {}
    for col, lag, score in ranked:
        src = layout.column_to_series[col]
        if src == target.series_id:
            continue
        curr = best_per_source.get(src)
        if curr is None or score > curr[1]:
            best_per_source[src] = (lag, score, col)

    out: list[CandidateEdge] = []
    for src, (lag, score, col) in best_per_source.items():
        if score < importance_threshold:
            continue
        out.append(
            CandidateEdge(
                source_series=src,
                target_series=target.series_id,
                lag=lag,
                importance=float(score),
                feature_column=col,
            )
        )
    ranked_for_debug = [(f"{col}__lag{lag}", float(score)) for col, lag, score in ranked]
    return out, last_design, ranked_for_debug


__all__ = ["CandidateEdge", "propose_candidates"]
