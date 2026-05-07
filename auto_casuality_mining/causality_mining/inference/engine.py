"""End-to-end causal inference: graph + new event -> per-target predictions.

Pipeline:
  1. Build a feature panel from `history` using the same `Change` encoder used
     during curation (backward 1-step change for every non-`pre_normalized`
     series).
  2. Resolve the new event into a scalar magnitude on the source node (a
     1-step change).
  3. Enumerate paths from the source node to every target node.
  4. For each path, refresh the per-edge effect against recent history (or
     reuse the cached `Edge.strength`) and multiply along the path. The refresh
     fits `forward(target, lag) ~ feature(source)` to match curation semantics.
  5. Aggregate paths into a per-target delta and confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causality_mining.curation.forward_target import forward_target_column
from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.edge import Edge
from causality_mining.graph.node import Node
from causality_mining.inference.effect import EdgeEffect, estimate_edge_effect
from causality_mining.inference.event import NewEvent
from causality_mining.inference.prediction import Prediction, TargetPrediction
from causality_mining.inference.propagate import propagate_event
from causality_mining.normalize.base import Change
from causality_mining.normalize.encode import encode_features
from causality_mining.panel.builder import build_panel
from causality_mining.timeseries.collection import TimeSeriesCollection
from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass
class InferenceConfig:
    """Knobs that mirror `DiscoveryConfig` so the panel rebuilds the same way."""

    freq: str = "1h"
    change: Change | None = None
    refresh_effects: bool = False
    max_depth: int = 1


def _event_magnitude(event: NewEvent, node: Node) -> float:
    """Reduce the event payload to a scalar 1-step-change treatment magnitude.

    `event.value` is interpreted as the change to inject (in the encoder's
    units), NOT a new raw level. This matches curation semantics where every
    edge is "per 1-step backward change of the source".

    - SCALAR: pass through as float.
    - VECTOR: average of the per-column changes when given as a dict, else cast.
    - CATEGORICAL: 1.0 if the value is non-empty/non-zero, else 0.0 (since
      a categorical "change" indicator has no further magnitude information).
    """
    if node.kind is TimeSeriesKind.CATEGORICAL:
        return 0.0 if not event.value else 1.0
    if node.kind is TimeSeriesKind.VECTOR and isinstance(event.value, dict):
        return float(np.mean(list(event.value.values())))
    return float(event.value)


def _last_raw_value(history: TimeSeriesCollection, raw_id: str) -> float:
    """Last non-NaN value of `raw_id` in the original (un-encoded) history."""
    if raw_id not in history:
        return 0.0
    ts = history.get(raw_id)
    data = ts.data
    if isinstance(data, pd.DataFrame):
        last = data.dropna(how="all")
        if last.empty:
            return 0.0
        return float(last.iloc[-1].mean())
    last = data.dropna()
    return float(last.iloc[-1]) if not last.empty else 0.0


def _path_effect(
    path: tuple[Edge, ...],
    panel: pd.DataFrame,
    layout_columns: dict[str, list[str]],
    encoded_to_raw: dict[str, TimeSeries],
    cfg: InferenceConfig,
) -> tuple[float, list[EdgeEffect], float]:
    """Multiply per-unit effects along a path; return (effect, edge_effects, conf)."""
    total = 1.0
    edge_effects: list[EdgeEffect] = []
    confidences: list[float] = []
    for edge in path:
        if cfg.refresh_effects and cfg.change is not None:
            ee = _refresh_edge_effect(edge, panel, layout_columns, encoded_to_raw, cfg)
        else:
            ee = EdgeEffect(edge=edge, per_unit=edge.strength, sample_size=0)
        edge_effects.append(ee)
        total *= ee.per_unit
        confidences.append(edge.confidence)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return total, edge_effects, confidence


def _refresh_edge_effect(
    edge: Edge,
    panel: pd.DataFrame,
    layout_columns: dict[str, list[str]],
    encoded_to_raw: dict[str, TimeSeries],
    cfg: InferenceConfig,
) -> EdgeEffect:
    """Re-estimate `edge` by fitting forward target change on backward source change."""
    source_cols = layout_columns.get(edge.source, [])
    raw_target = encoded_to_raw.get(edge.target)
    if not source_cols or raw_target is None or cfg.change is None:
        return EdgeEffect(edge=edge, per_unit=edge.strength, sample_size=0)
    treatment_col = source_cols[0]
    target_series = forward_target_column(raw_target, cfg.change, cfg.freq, edge.lag)
    outcome_col = "__outcome__"
    design = pd.concat(
        [panel[[treatment_col]], target_series.rename(outcome_col)],
        axis=1,
    ).dropna()
    return estimate_edge_effect(edge, design, treatment_col, outcome_col)


def predict(
    graph: CausalGraph,
    event: NewEvent,
    history: TimeSeriesCollection,
    config: InferenceConfig | None = None,
) -> Prediction:
    """Predict each target's response to `event` given recent `history`."""
    cfg = config or InferenceConfig()

    feature_collection = (
        encode_features(history, cfg.change) if cfg.change is not None else history
    )
    panel, layout = build_panel(feature_collection, cfg.freq)
    layout_columns = {sid: layout.series_columns(sid) for sid in graph.nodes}
    encoded_to_raw = {
        encoded.id: raw for raw, encoded in zip(history, feature_collection)
    }

    if event.series_id not in graph.nodes:
        return Prediction(event_series_id=event.series_id, targets={})
    source_node = graph.nodes[event.series_id]
    magnitude = _event_magnitude(event, source_node)

    paths_by_target = propagate_event(graph, event.series_id, max_depth=cfg.max_depth)
    out: dict[str, TargetPrediction] = {}
    for target_id, paths in paths_by_target.items():
        raw_target = encoded_to_raw.get(target_id)
        baseline = _last_raw_value(history, raw_target.id) if raw_target is not None else 0.0
        total_delta = 0.0
        all_edges: list[EdgeEffect] = []
        confidences: list[float] = []
        for path in paths:
            effect, edge_effects, path_conf = _path_effect(
                path, panel, layout_columns, encoded_to_raw, cfg
            )
            total_delta += effect * magnitude
            all_edges.extend(edge_effects)
            confidences.append(path_conf)
        if not all_edges:
            continue
        predicted = _predicted_level(
            baseline=baseline,
            delta=float(total_delta),
            raw_target=raw_target,
            change=cfg.change,
        )
        out[target_id] = TargetPrediction(
            target_id=target_id,
            delta=float(total_delta),
            baseline=baseline,
            predicted=predicted,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            contributing_edges=tuple(all_edges),
        )

    return Prediction(event_series_id=event.series_id, targets=out)


def _predicted_level(
    baseline: float,
    delta: float,
    raw_target: TimeSeries | None,
    change: Change | None,
) -> float:
    """Combine `baseline` (raw level) with `delta` (in encoder units) into a level.

    `pre_normalized` targets bypass the encoder: their "raw" series is already
    in change semantics, so additive combination is the natural fallback.
    Otherwise we ask the configured `Change` how to combine baseline with delta
    (e.g. PctChange returns baseline * (1 + delta)).
    """
    if change is None or (raw_target is not None and raw_target.pre_normalized):
        return float(baseline) + float(delta)
    return change.apply(float(baseline), float(delta))
