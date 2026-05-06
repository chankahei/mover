"""JSON serialization for `CausalGraph`.

The graph is the artifact the architecture doc's "weekly graph promotion and
rollback workflow" deals with, so it has to be portable.
"""
from __future__ import annotations

import json
from typing import Any

from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.edge import Edge
from causality_mining.graph.node import Node
from causality_mining.timeseries.kind import TimeSeriesKind


def to_dict(graph: CausalGraph) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": n.id, "kind": n.kind.value, "is_target": n.is_target}
            for n in graph.nodes.values()
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "lag": e.lag,
                "strength": e.strength,
                "confidence": e.confidence,
                "importance": e.importance,
                "refutations": e.refutations,
                "confounders": list(e.confounders),
                "meta": e.meta,
            }
            for e in graph.edges.values()
        ],
    }


def from_dict(payload: dict[str, Any]) -> CausalGraph:
    nodes = [
        Node(id=n["id"], kind=TimeSeriesKind(n["kind"]), is_target=bool(n.get("is_target", False)))
        for n in payload.get("nodes", [])
    ]
    edges = [
        Edge(
            source=e["source"],
            target=e["target"],
            lag=int(e["lag"]),
            strength=float(e["strength"]),
            confidence=float(e["confidence"]),
            importance=float(e.get("importance", 0.0)),
            refutations=dict(e.get("refutations", {})),
            confounders=tuple(e.get("confounders", ())),
            meta=dict(e.get("meta", {})),
        )
        for e in payload.get("edges", [])
    ]
    return CausalGraph.from_components(nodes, edges)


def save_json(graph: CausalGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(graph), f, indent=2)


def load_json(path: str) -> CausalGraph:
    with open(path, "r", encoding="utf-8") as f:
        return from_dict(json.load(f))
