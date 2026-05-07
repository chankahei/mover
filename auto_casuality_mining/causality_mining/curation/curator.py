"""End-to-end causal graph curation.

`curate_graph(collection, targets)` runs the full pipeline:

  1. Apply `cfg.change.backward` to every non-`pre_normalized` series to build
     a feature collection (each series carrying its 1-step backward change).
  2. Resample everything onto a common time grid -> panel DataFrame.
  3. For each target, propose candidate edges per lag using forward `lag`-step
     change of the (raw) target as the prediction signal.
  4. Refute each candidate with DoWhy and score robustness.
  5. Promote only edges whose confidence clears the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Sequence

import pandas as pd

from causality_mining.curation.candidates import propose_candidates
from causality_mining.curation.forward_target import forward_target_column
from causality_mining.curation.refute import refute_candidate, to_edge
from causality_mining.curation.targets import TargetSpec
from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.node import Node
from causality_mining.normalize.base import Change
from causality_mining.normalize.delta import Delta
from causality_mining.normalize.encode import encode_features
from causality_mining.panel.builder import build_panel
from causality_mining.timeseries.collection import TimeSeriesCollection
from causality_mining.timeseries.series import TimeSeries


@dataclass
class CurationConfig:
    """Knobs for the curation pipeline.

    `change` is the single change-encoder applied to features (backward, h=1)
    and to targets (forward, h=lag). Defaults to `Delta()`.
    """

    freq: str = "1h"
    change: Change = field(default_factory=Delta)
    importance_threshold: float = 0.01
    confidence_threshold: float = 0.4
    use_shap: bool = True
    debug: bool = False
    debug_top_features: int = 12


def curate_graph(
    collection: TimeSeriesCollection,
    targets: Sequence[TargetSpec],
    config: CurationConfig | None = None,
) -> CausalGraph:
    """Produce a refuted causal graph from a heterogeneous timeseries collection."""
    cfg = config or CurationConfig()
    debug_on = cfg.debug or os.environ.get("CAUSALITY_DEBUG") == "1"

    feature_collection = encode_features(collection, cfg.change)
    panel, layout = build_panel(feature_collection, cfg.freq)

    raw_by_id = {ts.id: ts for ts in collection}
    encoded_by_raw_id = {raw.id: enc for raw, enc in zip(collection, feature_collection)}

    graph = CausalGraph()
    for raw_ts, encoded_ts in zip(collection, feature_collection):
        is_target = any(t.series_id == raw_ts.id for t in targets)
        graph.add_node(Node(id=encoded_ts.id, kind=encoded_ts.kind, is_target=is_target))

    if debug_on:
        print(f"[debug] panel shape={panel.shape} freq={cfg.freq}")
        if not panel.empty:
            print(f"[debug] panel index range: {panel.index.min()} -> {panel.index.max()}")
        print(f"[debug] graph nodes from timeseries ids: {list(graph.nodes.keys())}")
        for sid in graph.nodes:
            cols = layout.series_columns(sid)
            preview = ", ".join(cols[:4])
            suffix = " ..." if len(cols) > 4 else ""
            print(f"[debug] series={sid:<8} columns={len(cols):<4} [{preview}{suffix}]")

    for target in targets:
        if target.series_id not in raw_by_id:
            continue
        raw_target_ts = raw_by_id[target.series_id]
        encoded_target_id = encoded_by_raw_id[target.series_id].id
        candidates, _last_design, ranked_importances = propose_candidates(
            panel=panel,
            layout=layout,
            target=_with_encoded_target_id(target, encoded_target_id),
            raw_target_ts=raw_target_ts,
            change=cfg.change,
            freq=cfg.freq,
            importance_threshold=cfg.importance_threshold,
            use_shap=cfg.use_shap,
        )
        if debug_on:
            print(
                f"[debug] target={encoded_target_id} feature_cols={len(panel.columns) - 1} "
                f"candidates_over_imp>={cfg.importance_threshold}: "
                f"{len(candidates)} / {len(ranked_importances)}"
            )
            for col, score in ranked_importances[: cfg.debug_top_features]:
                print(f"[debug] importance {score:+.6f}  feature={col}")
        _refute_and_promote(
            candidates=candidates,
            panel=panel,
            raw_target_ts=raw_target_ts,
            cfg=cfg,
            graph=graph,
            debug_on=debug_on,
        )

    if debug_on:
        print(f"[debug] final graph: nodes={len(graph.nodes)} edges={len(graph.edges)}")
    return graph


def _with_encoded_target_id(target: TargetSpec, encoded_id: str) -> TargetSpec:
    """Return a TargetSpec whose `series_id` matches the encoded panel node id."""
    return TargetSpec(
        series_id=encoded_id,
        kind=target.kind,
        max_lag=target.max_lag,
        min_lag=target.min_lag,
        extras=target.extras,
    )


def _refute_and_promote(
    candidates,
    panel: pd.DataFrame,
    raw_target_ts: TimeSeries,
    cfg: CurationConfig,
    graph: CausalGraph,
    debug_on: bool,
) -> None:
    """Run DoWhy refutation on each candidate and add passing edges to `graph`.

    Builds a per-candidate design (treatment column at time t + forward
    `lag`-step change of the target at time t) so refutation matches the lag
    each candidate was discovered at.
    """
    outcome_column = "__outcome__"
    for candidate in candidates:
        target_series = forward_target_column(
            raw_target_ts, cfg.change, cfg.freq, candidate.lag
        )
        design = pd.concat(
            [panel[[candidate.feature_column]], target_series.rename(outcome_column)],
            axis=1,
        ).dropna()
        if design.empty:
            if debug_on:
                print(
                    f"[debug] DROP {candidate.source_series}->{candidate.target_series} "
                    f"lag={candidate.lag} (empty design)"
                )
            continue
        refutation = refute_candidate(
            candidate=candidate,
            panel=design,
            treatment_column=candidate.feature_column,
            outcome_column=outcome_column,
        )
        if refutation is None:
            if debug_on:
                print(
                    f"[debug] DROP {candidate.source_series}->{candidate.target_series} "
                    f"lag={candidate.lag} (refute failed/insufficient variation)"
                )
            continue
        edge = to_edge(candidate, refutation)
        if debug_on:
            print(
                f"[debug] candidate {edge.source}->{edge.target} lag={edge.lag} "
                f"imp={edge.importance:+.6f} est={edge.strength:+.6f} "
                f"conf={edge.confidence:.3f} refuters={edge.refutations}"
            )
        if edge.confidence < cfg.confidence_threshold:
            if debug_on:
                print(
                    f"[debug] DROP {edge.source}->{edge.target} lag={edge.lag} "
                    f"confidence {edge.confidence:.3f} < {cfg.confidence_threshold:.3f}"
                )
            continue
        if debug_on:
            print(f"[debug] KEEP {edge.source}->{edge.target} lag={edge.lag}")
        graph.add_edge(edge)
