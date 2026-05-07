"""Leave-one-out causal discovery on a `TimeSeriesCollection`.

Causal claim of an edge `A -> B` with lag `L`:

    "A 1-step backward change of A at time t causes the L-step forward change
    of B between t and t+L."

Concretely:

- Features: backward 1-step change of every series at time t, via
  `cfg.change.backward(ts, 1)`. Each series contributes one (or, for VECTORs,
  one pooled) panel column.
- Per-target loop: for every non-VECTOR, non-CATEGORICAL target series, fit
  ONE LightGBM model PER LAG predicting `cfg.change.forward(target, lag)` at
  time t from the feature columns at time t. Per-feature SHAP gives the
  per-source importance/strength at that lag.
- Aggregation: the (source, lag) with highest importance is kept per source,
  then thresholds + (optional) DoWhy refutation decide which become edges.

VECTOR sources are reduced to a single panel column via sign-preserving
abs-max pooling. VECTOR / CATEGORICAL targets are skipped (they would yield
multi-output regressions which this module does not implement).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np
import pandas as pd

from causality_mining.curation.candidates import CandidateEdge
from causality_mining.curation.forward_target import forward_target_column
from causality_mining.curation.loo import FeatureScore, fit_one_lag
from causality_mining.curation.refute import refute_candidate, to_edge
from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.edge import Edge
from causality_mining.graph.node import Node
from causality_mining.normalize.base import Change
from causality_mining.normalize.delta import Delta
from causality_mining.normalize.encode import encode_features
from causality_mining.panel.builder import ColumnLayout
from causality_mining.panel.resample import resample_to_grid
from causality_mining.timeseries.collection import TimeSeriesCollection
from causality_mining.timeseries.kind import TimeSeriesKind


_DEFAULT_LAGS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass
class DiscoveryConfig:
    """Knobs for `discover_graph`.

    `change` is the single change-encoder applied to features (backward, h=1)
    and to targets (forward, h=lag). Defaults to `Delta()`. To use multiplicative
    change, pass `PctChange()`; to use log-additive change, pass `LogReturn()`.
    """

    freq: str = "1D"
    change: Change = field(default_factory=Delta)
    lags: tuple[int, ...] = _DEFAULT_LAGS
    importance_threshold: float = 0.0
    min_confidence: float = 0.0
    max_parents_per_target: int | None = None
    dowhy_refute_top_k_per_target: int = 0
    dowhy_confidence_threshold: float = 0.0
    debug: bool = False
    debug_top_features: int = 10


def discover_graph(
    collection: TimeSeriesCollection,
    config: DiscoveryConfig | None = None,
) -> CausalGraph:
    """Build a causal graph by predicting forward target change from backward feature change."""
    cfg = config or DiscoveryConfig()
    debug_on = cfg.debug or os.environ.get("CAUSALITY_DEBUG") == "1"

    feature_collection = encode_features(collection, cfg.change)
    feature_panel, layout = _build_feature_panel(feature_collection, cfg)

    graph = CausalGraph()
    pairs = list(zip(collection, feature_collection))
    for raw_ts, encoded_ts in pairs:
        graph.add_node(
            Node(
                id=encoded_ts.id,
                kind=encoded_ts.kind,
                is_target=raw_ts.kind is TimeSeriesKind.SCALAR,
            )
        )

    if debug_on:
        print(f"[discover] panel shape={feature_panel.shape} freq={cfg.freq} lags={cfg.lags}")
        for raw_ts, encoded_ts in pairs:
            cols = layout.series_columns(encoded_ts.id)
            print(
                f"[discover] series={encoded_ts.id} raw_kind={raw_ts.kind.value} "
                f"feature_cols={len(cols)}"
            )

    for raw_ts, encoded_ts in pairs:
        if raw_ts.kind is not TimeSeriesKind.SCALAR:
            if debug_on:
                print(f"[discover] skip target {encoded_ts.id} (raw_kind={raw_ts.kind.value})")
            continue
        _discover_one_target(
            encoded_target_id=encoded_ts.id,
            raw_target_ts=raw_ts,
            raw_target_kind=raw_ts.kind,
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
    encoded_target_id: str,
    raw_target_ts,
    raw_target_kind: TimeSeriesKind,
    feature_panel: pd.DataFrame,
    layout: ColumnLayout,
    cfg: DiscoveryConfig,
    graph: CausalGraph,
    debug_on: bool,
) -> None:
    """Fit one model per lag, aggregate per source, add edges to `graph`."""
    feature_cols = [
        c for c in feature_panel.columns if layout.column_to_series[c] != encoded_target_id
    ]
    if not feature_cols:
        return

    per_lag_scores, n_samples = _fit_all_lags(
        raw_target_ts=raw_target_ts,
        raw_target_kind=raw_target_kind,
        feature_panel=feature_panel,
        feature_cols=feature_cols,
        cfg=cfg,
    )
    if debug_on:
        print(
            f"[discover] target={encoded_target_id} kind={raw_target_kind.value} "
            f"n_samples={n_samples} feature_cols={len(feature_cols)} "
            f"lags_with_scores={sum(1 for s in per_lag_scores.values() if s)}"
        )
    if not per_lag_scores:
        return

    per_source_lag = _aggregate_per_source_lag(per_lag_scores, layout, encoded_target_id)
    per_source_best = _best_lag_per_source(per_source_lag)

    if debug_on:
        ranked = sorted(per_source_best.items(), key=lambda kv: -kv[1][1])
        for src, (lag, imp, strength) in ranked[: cfg.debug_top_features]:
            print(
                f"[discover] {src} -> {encoded_target_id} best_lag={lag} "
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
                    f"[discover] DROP {src}->{encoded_target_id} lag={lag} "
                    f"imp={imp:.6f} (< importance_threshold={cfg.importance_threshold})"
                )
            continue
        confidence = float(imp / max_imp) if max_imp > 0 else 0.0
        if confidence < cfg.min_confidence:
            n_drop_conf += 1
            if debug_on:
                print(
                    f"[discover] DROP {src}->{encoded_target_id} lag={lag} "
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
                target=encoded_target_id,
                lag=lag,
                strength=strength,
                confidence=confidence,
                importance=imp,
            ),
            src_cols[0],
        ))

    ranked_edges.sort(key=lambda item: item[0].importance, reverse=True)
    n_pre_topk = len(ranked_edges)
    if cfg.max_parents_per_target is not None:
        ranked_edges = ranked_edges[: max(0, cfg.max_parents_per_target)]
    n_post_topk = len(ranked_edges)
    n_drop_topk = max(0, n_pre_topk - n_post_topk)

    final_edges = _apply_dowhy_refutation(
        ranked_edges=ranked_edges,
        encoded_target_id=encoded_target_id,
        raw_target_ts=raw_target_ts,
        feature_panel=feature_panel,
        cfg=cfg,
        debug_on=debug_on,
    )
    n_drop_dowhy = max(0, n_post_topk - len(final_edges))
    for edge in final_edges:
        graph.add_edge(edge)

    if debug_on:
        print(
            f"[discover] funnel for {encoded_target_id}: "
            f"sources={n_total} -> after_imp={n_total - n_drop_imp} "
            f"-> after_relconf={n_pre_topk} (drop_imp={n_drop_imp}, drop_relconf={n_drop_conf}) "
            f"-> after_topk={n_post_topk} (drop_topk={n_drop_topk}) "
            f"-> after_dowhy={len(final_edges)} (drop_dowhy={n_drop_dowhy})"
        )
        print(
            f"[discover] kept {len(final_edges)} parents for {encoded_target_id} "
            f"(min_imp={cfg.importance_threshold}, min_conf={cfg.min_confidence}, "
            f"max_parents={cfg.max_parents_per_target}, dowhy_topk={cfg.dowhy_refute_top_k_per_target})"
        )


def _fit_all_lags(
    raw_target_ts,
    raw_target_kind: TimeSeriesKind,
    feature_panel: pd.DataFrame,
    feature_cols: list[str],
    cfg: DiscoveryConfig,
) -> tuple[dict[int, dict[str, FeatureScore]], int]:
    """Run `fit_one_lag` for every lag in `cfg.lags`; return scores keyed by lag."""
    per_lag: dict[int, dict[str, FeatureScore]] = {}
    last_n_samples = 0
    for lag in cfg.lags:
        target_series = forward_target_column(raw_target_ts, cfg.change, cfg.freq, lag)
        scores, n = fit_one_lag(
            feature_panel=feature_panel,
            target_series=target_series,
            target_kind=raw_target_kind,
            feature_cols=feature_cols,
        )
        per_lag[lag] = scores
        last_n_samples = n
    return per_lag, last_n_samples


def _aggregate_per_source_lag(
    per_lag_scores: dict[int, dict[str, FeatureScore]],
    layout: ColumnLayout,
    encoded_target_id: str,
) -> dict[tuple[str, int], tuple[float, float]]:
    """Sum per-column SHAP up to `(source_series, lag)`."""
    bucket: dict[tuple[str, int], list[FeatureScore]] = {}
    for lag, scores in per_lag_scores.items():
        for col, score in scores.items():
            src = layout.column_to_series.get(col)
            if src is None or src == encoded_target_id:
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
    per_source_lag: dict[tuple[str, int], tuple[float, float]],
) -> dict[str, tuple[int, float, float]]:
    """For each source, keep the `(lag, importance, strength)` with max importance."""
    best: dict[str, tuple[int, float, float]] = {}
    for (src, lag), (imp, strength) in per_source_lag.items():
        curr = best.get(src)
        if curr is None or imp > curr[1]:
            best[src] = (lag, imp, strength)
    return best


def _build_feature_panel(
    feature_collection: TimeSeriesCollection,
    cfg: DiscoveryConfig,
) -> tuple[pd.DataFrame, ColumnLayout]:
    """Resample each (already backward-changed) series to `cfg.freq` and concatenate.

    VECTOR series collapse to a single column via sign-preserving abs-max
    pooling along the column axis. Semantically: any column firing implies the
    vector as a whole is firing.
    """
    frames: list[pd.DataFrame] = []
    column_to_series: dict[str, str] = {}
    for ts in feature_collection:
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
    encoded_target_id: str,
    raw_target_ts,
    feature_panel: pd.DataFrame,
    cfg: DiscoveryConfig,
    debug_on: bool,
) -> list[Edge]:
    """Optionally refute top-k discovered edges with DoWhy and filter by confidence."""
    if cfg.dowhy_refute_top_k_per_target <= 0:
        return [edge for edge, _ in ranked_edges]
    if not ranked_edges:
        return []

    refute_k = min(cfg.dowhy_refute_top_k_per_target, len(ranked_edges))
    out: list[Edge] = []
    for idx, (edge, treatment_col) in enumerate(ranked_edges):
        if idx >= refute_k:
            out.append(edge)
            continue
        target_series = forward_target_column(raw_target_ts, cfg.change, cfg.freq, edge.lag)
        outcome_col = f"__outcome_lag{edge.lag}__"
        design = pd.concat(
            [feature_panel[[treatment_col]], target_series.rename(outcome_col)],
            axis=1,
        ).dropna()
        candidate = CandidateEdge(
            source_series=edge.source,
            target_series=encoded_target_id,
            lag=edge.lag,
            importance=edge.importance,
            feature_column=treatment_col,
        )
        refutation = refute_candidate(
            candidate=candidate,
            panel=design,
            treatment_column=treatment_col,
            outcome_column=outcome_col,
        )
        if refutation is None:
            if debug_on:
                print(
                    f"[discover] DROP {edge.source}->{encoded_target_id} lag={edge.lag} "
                    f"(DoWhy failed/insufficient data)"
                )
            continue
        refuted_edge = to_edge(candidate, refutation)
        if debug_on:
            score_str = ", ".join(
                f"{name}={value:.3f}" for name, value in refutation.scores.items()
            )
            print(
                f"[discover] DOWHY {edge.source}->{encoded_target_id} lag={edge.lag} "
                f"est={refutation.estimate:+.4f} mean_conf={refuted_edge.confidence:.3f} "
                f"scores=[{score_str}]"
            )
        if refuted_edge.confidence < cfg.dowhy_confidence_threshold:
            if debug_on:
                print(
                    f"[discover] DROP {edge.source}->{encoded_target_id} lag={edge.lag} "
                    f"(DoWhy conf={refuted_edge.confidence:.3f} < {cfg.dowhy_confidence_threshold:.3f})"
                )
            continue
        out.append(refuted_edge)
    return out
