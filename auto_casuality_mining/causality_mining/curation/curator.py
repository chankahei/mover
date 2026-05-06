"""End-to-end causal graph curation.

`curate_graph(collection, targets)` runs the full pipeline described in the
architecture doc's "Causal Graph Discovery and Refutation Pipeline":

  1. Optionally normalize the collection (delta is on by default).
  2. Resample everything onto a common time grid -> panel DataFrame.
  3. For each target, propose candidate edges via lagged predictive models.
  4. Refute each candidate with DoWhy and score robustness.
  5. Promote only edges whose confidence clears the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Sequence

from causality_mining.curation.candidates import propose_candidates
from causality_mining.curation.refute import refute_candidate, to_edge
from causality_mining.curation.targets import TargetSpec
from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.node import Node
from causality_mining.normalize.delta import Delta
from causality_mining.normalize.pipeline import Pipeline
from causality_mining.panel.builder import build_panel
from causality_mining.timeseries.collection import TimeSeriesCollection


@dataclass
class CurationConfig:
    """Knobs for the curation pipeline.

    Defaults are deliberately conservative: only edges with non-trivial
    importance AND non-trivial post-refutation confidence are promoted.
    """

    freq: str = "1h"
    normalize: Pipeline = field(default_factory=lambda: Pipeline(steps=(Delta(),)))
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

    normalized = cfg.normalize.apply_collection(collection) if cfg.normalize.steps else collection
    panel, layout = build_panel(normalized, cfg.freq)
    debug_on = cfg.debug or os.environ.get("CAUSALITY_DEBUG") == "1"

    graph = CausalGraph()
    for ts in normalized:
        is_target = any(t.series_id == ts.id for t in targets)
        graph.add_node(Node(id=ts.id, kind=ts.kind, is_target=is_target))
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
        if target.series_id not in graph.nodes:
            continue
        target_columns = layout.series_columns(target.series_id)
        if not target_columns:
            continue
        candidates, design, ranked_importances = propose_candidates(
            panel=panel,
            layout=layout,
            target=target,
            target_columns=target_columns,
            importance_threshold=cfg.importance_threshold,
            use_shap=cfg.use_shap,
        )
        if debug_on:
            print(
                f"[debug] target={target.series_id} columns={target_columns} "
                f"design_shape={design.shape} lagged_features={max(len(design.columns) - 1, 0)}"
            )
            print(
                f"[debug] candidates over importance>={cfg.importance_threshold}: "
                f"{len(candidates)} / {len(ranked_importances)}"
            )
            for col, score in ranked_importances[: cfg.debug_top_features]:
                print(f"[debug] importance {score:+.6f}  feature={col}")
        for candidate in candidates:
            refutation = refute_candidate(
                candidate=candidate,
                panel=design,
                treatment_column=candidate.feature_column,
                outcome_column=target_columns[0],
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
                    f"imp={edge.importance:+.6f} est={edge.strength:+.6f} conf={edge.confidence:.3f} "
                    f"refuters={edge.refutations}"
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

    if debug_on:
        print(f"[debug] final graph: nodes={len(graph.nodes)} edges={len(graph.edges)}")
    return graph
