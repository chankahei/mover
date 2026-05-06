"""Leave-one-out causal discovery on a `TimeSeriesCollection`.

For every SCALAR or CATEGORICAL series, fit a lagged linear / logistic model
that predicts its current value from every other series, then aggregate SHAP
importances per `(source_series, lag)` to propose edges. VECTOR series are
NEVER predicted as targets but remain available as input features.

Vector input features are reduced to a single scalar via element-wise max
pooling along the column axis (sign-preserving abs-max): if any element of the
vector has a causal effect at a given lag, the whole vector is treated as
having a causal effect.

Public API: `discover_graph(collection, config)` and `DiscoveryConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np
import pandas as pd

from causality_mining.curation.candidates import CandidateEdge
from causality_mining.curation.loo import FeatureScore, fit_loo_target
from causality_mining.curation.refute import refute_candidate, to_edge
from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.edge import Edge
from causality_mining.graph.node import Node
from causality_mining.normalize.delta import Delta
from causality_mining.normalize.pipeline import Pipeline
from causality_mining.panel.builder import ColumnLayout
from causality_mining.panel.lag import lagged_columns
from causality_mining.panel.resample import resample_to_grid
from causality_mining.timeseries.collection import TimeSeriesCollection
from causality_mining.timeseries.kind import TimeSeriesKind


_DEFAULT_LAGS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass
class DiscoveryConfig:
    """Knobs for `discover_graph`.

    Defaults match the leave-one-out spec: lags `(1, 2, 4, 8, 16, 32)`, Ridge /
    Logistic regression with `alpha=1.0`, and VECTOR input features reduced to
    a single scalar per timestamp via sign-preserving abs-max pooling.
    """

    freq: str = "1D"
    normalize: Pipeline = field(default_factory=lambda: Pipeline(steps=(Delta(),)))
    lags: tuple[int, ...] = _DEFAULT_LAGS
    importance_threshold: float = 0.0
    min_confidence: float = 0.0
    max_parents_per_target: int | None = None
    dowhy_refute_top_k_per_target: int = 0
    dowhy_confidence_threshold: float = 0.0
    alpha: float = 1.0
    debug: bool = False
    debug_top_features: int = 10


def discover_graph(
    collection: TimeSeriesCollection,
    config: DiscoveryConfig | None = None,
) -> CausalGraph:
    """Build a causal graph by predicting every non-VECTOR series from all others."""
    cfg = config or DiscoveryConfig()
    debug_on = cfg.debug or os.environ.get("CAUSALITY_DEBUG") == "1"

    normalized = (
        cfg.normalize.apply_collection(collection) if cfg.normalize.steps else collection
    )

    feature_panel, layout = _build_feature_panel(normalized, cfg)

    graph = CausalGraph()
    series_kind: dict[str, TimeSeriesKind] = {}
    for ts in normalized:
        graph.add_node(Node(id=ts.id, kind=ts.kind, is_target=ts.kind is not TimeSeriesKind.VECTOR))
        series_kind[ts.id] = ts.kind

    if debug_on:
        print(f"[discover] panel shape={feature_panel.shape} freq={cfg.freq} lags={cfg.lags}")
        for sid, kind in series_kind.items():
            cols = layout.series_columns(sid)
            print(f"[discover] series={sid} kind={kind.value} feature_cols={len(cols)}")

    for target_series_id, target_kind in series_kind.items():
        if target_kind is TimeSeriesKind.VECTOR:
            if debug_on:
                print(f"[discover] skip target {target_series_id} (VECTOR)")
            continue
        _discover_one_target(
            target_series_id=target_series_id,
            target_kind=target_kind,
            feature_panel=feature_panel,
            layout=layout,
            cfg=cfg,
            graph=graph,
            debug_on=debug_on,
        )

    if debug_on:
        print(f"[discover] final graph: nodes={len(graph.nodes)} edges={len(graph.edges)}")
    return graph


def _discover_one_target(
    target_series_id: str,
    target_kind: TimeSeriesKind,
    feature_panel: pd.DataFrame,
    layout: ColumnLayout,
    cfg: DiscoveryConfig,
    graph: CausalGraph,
    debug_on: bool,
) -> None:
    """Fit one LOO target and add its passing edges to `graph` in place."""
    target_cols = layout.series_columns(target_series_id)
    if not target_cols:
        return
    target_col = target_cols[0]
    feature_cols = [
        c for c in feature_panel.columns if layout.column_to_series[c] != target_series_id
    ]

    scores, n_samples = fit_loo_target(
        panel=feature_panel,
        target_col=target_col,
        target_kind=target_kind,
        feature_cols=feature_cols,
        lags=cfg.lags,
        alpha=cfg.alpha,
    )
    if debug_on:
        print(
            f"[discover] target={target_series_id} kind={target_kind.value} "
            f"n_samples={n_samples} feature_cols={len(feature_cols)} scores={len(scores)}"
        )
    if not scores:
        return

    per_lag = _aggregate_per_source_lag(scores, layout, target_series_id)
    per_source_best = _best_lag_per_source(per_lag)

    if debug_on:
        ranked = sorted(per_source_best.items(), key=lambda kv: -kv[1][1])
        for src, (lag, imp, strength) in ranked[: cfg.debug_top_features]:
            print(
                f"[discover] {src} -> {target_series_id} best_lag={lag} "
                f"imp={imp:+.6f} strength={strength:+.6f}"
            )

    max_imp = max((imp for _, imp, _ in per_source_best.values()), default=0.0)
    ranked_edges: list[tuple[Edge, str]] = []
    n_total = len(per_source_best)
    n_drop_imp = 0
    n_drop_conf = 0
    for src, (lag, imp, strength) in per_source_best.items():
        if imp < cfg.importance_threshold:
            n_drop_imp += 1
            if debug_on:
                print(
                    f"[discover] DROP {src}->{target_series_id} lag={lag} "
                    f"imp={imp:.6f} (< importance_threshold={cfg.importance_threshold})"
                )
            continue
        confidence = float(imp / max_imp) if max_imp > 0 else 0.0
        if confidence < cfg.min_confidence:
            n_drop_conf += 1
            if debug_on:
                print(
                    f"[discover] DROP {src}->{target_series_id} lag={lag} "
                    f"imp={imp:.6f} rel_conf={confidence:.3f} "
                    f"(< min_confidence={cfg.min_confidence})"
                )
            continue
        src_cols = layout.series_columns(src)
        if not src_cols:
            continue
        ranked_edges.append((
            Edge(
                source=src,
                target=target_series_id,
                lag=lag,
                strength=strength,
                confidence=confidence,
                importance=imp,
            ),
            f"{src_cols[0]}__lag{lag}",
        ))

    ranked_edges.sort(key=lambda item: item[0].importance, reverse=True)
    n_pre_topk = len(ranked_edges)
    if cfg.max_parents_per_target is not None:
        ranked_edges = ranked_edges[: max(0, cfg.max_parents_per_target)]
    n_post_topk = len(ranked_edges)
    n_drop_topk = max(0, n_pre_topk - n_post_topk)

    final_edges = _apply_dowhy_refutation(
        ranked_edges=ranked_edges,
        target_series_id=target_series_id,
        target_col=target_col,
        feature_panel=feature_panel,
        feature_cols=feature_cols,
        cfg=cfg,
        debug_on=debug_on,
    )
    n_drop_dowhy = max(0, n_post_topk - len(final_edges))
    for edge in final_edges:
        graph.add_edge(edge)

    if debug_on:
        print(
            f"[discover] funnel for {target_series_id}: "
            f"sources={n_total} -> after_imp={n_total - n_drop_imp} "
            f"-> after_relconf={n_pre_topk} (drop_imp={n_drop_imp}, drop_relconf={n_drop_conf}) "
            f"-> after_topk={n_post_topk} (drop_topk={n_drop_topk}) "
            f"-> after_dowhy={len(final_edges)} (drop_dowhy={n_drop_dowhy})"
        )
        print(
            f"[discover] kept {len(final_edges)} parents for {target_series_id} "
            f"(min_imp={cfg.importance_threshold}, min_conf={cfg.min_confidence}, "
            f"max_parents={cfg.max_parents_per_target}, dowhy_topk={cfg.dowhy_refute_top_k_per_target})"
        )


def _aggregate_per_source_lag(
    scores: dict[str, FeatureScore],
    layout: ColumnLayout,
    target_series_id: str,
) -> dict[tuple[str, int], tuple[float, float]]:
    """Sum per-column SHAP importance/strength up to `(source_series, lag)`."""
    bucket: dict[tuple[str, int], list[FeatureScore]] = {}
    for col, score in scores.items():
        base, lag_part = col.rsplit("__lag", 1)
        src = layout.column_to_series.get(base)
        if src is None or src == target_series_id:
            continue
        try:
            lag = int(lag_part)
        except ValueError:
            continue
        bucket.setdefault((src, lag), []).append(score)
    return {
        key: (
            float(sum(v.importance for v in vals)),
            float(sum(v.signed_strength for v in vals)),
        )
        for key, vals in bucket.items()
    }


def _best_lag_per_source(
    per_lag: dict[tuple[str, int], tuple[float, float]],
) -> dict[str, tuple[int, float, float]]:
    """For each source, keep the `(lag, importance, strength)` with max importance."""
    best: dict[str, tuple[int, float, float]] = {}
    for (src, lag), (imp, strength) in per_lag.items():
        curr = best.get(src)
        if curr is None or imp > curr[1]:
            best[src] = (lag, imp, strength)
    return best


def _build_feature_panel(
    collection: TimeSeriesCollection,
    cfg: DiscoveryConfig,
) -> tuple[pd.DataFrame, ColumnLayout]:
    """Resample each series to `cfg.freq` and concatenate.

    VECTOR series are always reduced to a single column per timestamp via
    sign-preserving abs-max pooling along the column axis. Semantically: if
    any element of the vector has a causal effect at a given lag, the vector
    as a whole is treated as having that causal effect.
    """
    frames: list[pd.DataFrame] = []
    column_to_series: dict[str, str] = {}
    for ts in collection:
        df = resample_to_grid(ts, cfg.freq)
        if ts.kind is TimeSeriesKind.VECTOR:
            df = _maxpool_vector(df, name=f"{ts.id}__maxpool")
        for col in df.columns:
            column_to_series[col] = ts.id
        frames.append(df)
    if not frames:
        return pd.DataFrame(), ColumnLayout({})
    panel = pd.concat(frames, axis=1).sort_index()
    return panel, ColumnLayout(column_to_series)


def _maxpool_vector(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Reduce a 2-D vector frame to a single column via sign-preserving abs-max.

    For each row, pick the element with the largest absolute value while keeping
    its original sign. NaN rows propagate.
    """
    if df.empty:
        return pd.DataFrame({name: pd.Series(dtype=float)}, index=df.index)
    values = df.to_numpy(dtype=float, copy=True)
    abs_values = np.abs(values)
    valid_mask = ~np.isnan(values)
    safe_abs = np.where(valid_mask, abs_values, -np.inf)
    has_value = valid_mask.any(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    if has_value.any():
        idx = np.argmax(safe_abs[has_value], axis=1)
        rows = np.flatnonzero(has_value)
        out[rows] = values[rows, idx]
    return pd.DataFrame({name: out}, index=df.index)


def _apply_dowhy_refutation(
    ranked_edges: list[tuple[Edge, str]],
    target_series_id: str,
    target_col: str,
    feature_panel: pd.DataFrame,
    feature_cols: list[str],
    cfg: DiscoveryConfig,
    debug_on: bool,
) -> list[Edge]:
    """Optionally refute top-k discovered edges with DoWhy and filter by confidence."""
    if cfg.dowhy_refute_top_k_per_target <= 0:
        return [edge for edge, _ in ranked_edges]
    if not ranked_edges:
        return []

    x_lagged = lagged_columns(feature_panel, feature_cols, cfg.lags)
    design = pd.concat([x_lagged, feature_panel[[target_col]]], axis=1).dropna()
    out: list[Edge] = []
    refute_k = min(cfg.dowhy_refute_top_k_per_target, len(ranked_edges))

    for idx, (edge, feature_column) in enumerate(ranked_edges):
        if idx >= refute_k:
            out.append(edge)
            continue
        candidate = CandidateEdge(
            source_series=edge.source,
            target_series=target_series_id,
            lag=edge.lag,
            importance=edge.importance,
            feature_column=feature_column,
        )
        refutation = refute_candidate(
            candidate=candidate,
            panel=design,
            treatment_column=feature_column,
            outcome_column=target_col,
        )
        if refutation is None:
            if debug_on:
                print(
                    f"[discover] DROP {edge.source}->{target_series_id} lag={edge.lag} "
                    f"(DoWhy failed/insufficient data)"
                )
            continue
        refuted_edge = to_edge(candidate, refutation)
        if debug_on:
            score_str = ", ".join(
                f"{name}={value:.3f}" for name, value in refutation.scores.items()
            )
            print(
                f"[discover] DOWHY {edge.source}->{target_series_id} lag={edge.lag} "
                f"est={refutation.estimate:+.4f} mean_conf={refuted_edge.confidence:.3f} "
                f"scores=[{score_str}]"
            )
        if refuted_edge.confidence < cfg.dowhy_confidence_threshold:
            if debug_on:
                print(
                    f"[discover] DROP {edge.source}->{target_series_id} lag={edge.lag} "
                    f"(DoWhy conf={refuted_edge.confidence:.3f} < {cfg.dowhy_confidence_threshold:.3f})"
                )
            continue
        out.append(refuted_edge)
    return out
