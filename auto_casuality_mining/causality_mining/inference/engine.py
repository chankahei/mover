"""End-to-end causal inference: graph + new event -> per-target predictions.

Pipeline:
  1. Resolve the new event into a scalar magnitude on the source node.
  2. Enumerate paths from the source node to every target node.
  3. For each path, refresh the per-edge effect against recent history (or
     reuse the cached `Edge.strength`) and multiply along the path.
  4. Aggregate paths into a per-target delta and confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.edge import Edge
from causality_mining.graph.node import Node
from causality_mining.inference.effect import EdgeEffect, estimate_edge_effect
from causality_mining.inference.event import NewEvent
from causality_mining.inference.prediction import Prediction, TargetPrediction
from causality_mining.inference.propagate import propagate_event
from causality_mining.normalize.delta import Delta
from causality_mining.normalize.pipeline import Pipeline
from causality_mining.panel.builder import build_panel
from causality_mining.panel.lag import lagged_columns
from causality_mining.timeseries.collection import TimeSeriesCollection
from causality_mining.timeseries.kind import TimeSeriesKind


@dataclass
class InferenceConfig:
    """Knobs that mirror `CurationConfig` so the panel rebuilds the same way."""

    freq: str = "1h"
    normalize: Pipeline | None = None
    refresh_effects: bool = False
    max_depth: int = 1


def _event_magnitude(event: NewEvent, node: Node, last_value: float) -> float:
    """Reduce a heterogeneous event value to a scalar treatment magnitude.

    - SCALAR / VECTOR: delta from the last observed level (matches the default
      Delta normalization used during curation).
    - CATEGORICAL: 1.0 if the level changed, else 0.0.
    """
    if node.kind is TimeSeriesKind.CATEGORICAL:
        return 0.0 if event.value == last_value else 1.0
    if node.kind is TimeSeriesKind.VECTOR and isinstance(event.value, dict):
        return float(np.mean(list(event.value.values())) - last_value)
    return float(event.value) - float(last_value)


def _last_value(panel: pd.DataFrame, columns: list[str]) -> float:
    if not columns or panel.empty:
        return 0.0
    last_row = panel[columns].dropna(how="all").iloc[-1]
    return float(last_row.mean())


def _path_effect(
    path: tuple[Edge, ...],
    panel: pd.DataFrame,
    layout_columns: dict[str, list[str]],
    refresh: bool,
) -> tuple[float, list[EdgeEffect], float]:
    """Multiply per-unit effects along a path; return (effect, edge_effects, conf)."""
    total = 1.0
    edge_effects: list[EdgeEffect] = []
    confidences: list[float] = []
    for edge in path:
        if refresh:
            source_cols = layout_columns.get(edge.source, [])
            target_cols = layout_columns.get(edge.target, [])
            if not source_cols or not target_cols:
                edge_effects.append(EdgeEffect(edge=edge, per_unit=edge.strength, sample_size=0))
                total *= edge.strength
                confidences.append(edge.confidence)
                continue
            lagged = lagged_columns(panel, [source_cols[0]], [edge.lag])
            treatment_col = f"{source_cols[0]}__lag{edge.lag}"
            design = pd.concat([lagged, panel[[target_cols[0]]]], axis=1).dropna()
            ee = estimate_edge_effect(edge, design, treatment_col, target_cols[0])
        else:
            ee = EdgeEffect(edge=edge, per_unit=edge.strength, sample_size=0)
        edge_effects.append(ee)
        total *= ee.per_unit
        confidences.append(edge.confidence)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return total, edge_effects, confidence


def predict(
    graph: CausalGraph,
    event: NewEvent,
    history: TimeSeriesCollection,
    config: InferenceConfig | None = None,
) -> Prediction:
    """Predict each target's response to `event` given recent `history`."""
    cfg = config or InferenceConfig()

    normalized = cfg.normalize.apply_collection(history) if (cfg.normalize and cfg.normalize.steps) else history
    panel, layout = build_panel(normalized, cfg.freq)
    layout_columns = {sid: layout.series_columns(sid) for sid in graph.nodes}

    if event.series_id not in graph.nodes:
        return Prediction(event_series_id=event.series_id, targets={})
    source_node = graph.nodes[event.series_id]
    source_columns = layout_columns.get(event.series_id, [])
    last = _last_value(panel, source_columns)
    magnitude = _event_magnitude(event, source_node, last)

    paths_by_target = propagate_event(graph, event.series_id, max_depth=cfg.max_depth)
    out: dict[str, TargetPrediction] = {}
    for target_id, paths in paths_by_target.items():
        target_columns = layout_columns.get(target_id, [])
        baseline = _last_value(panel, target_columns)
        total_delta = 0.0
        all_edges: list[EdgeEffect] = []
        confidences: list[float] = []
        for path in paths:
            effect, edge_effects, path_conf = _path_effect(
                path, panel, layout_columns, cfg.refresh_effects
            )
            total_delta += effect * magnitude
            all_edges.extend(edge_effects)
            confidences.append(path_conf)
        if not all_edges:
            continue
        out[target_id] = TargetPrediction(
            target_id=target_id,
            delta=float(total_delta),
            baseline=baseline,
            predicted=baseline + float(total_delta),
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            contributing_edges=tuple(all_edges),
        )

    return Prediction(event_series_id=event.series_id, targets=out)
