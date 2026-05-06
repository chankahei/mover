"""Walk source -> target paths in a `CausalGraph` and accumulate effects.

Two styles are supported:

- **Direct edges only** (default, fast): each target is reached via at most one
  edge from the event source. The total delta is simply `effect * scalar`.
- **Multi-hop**: the engine recurses through intermediate nodes, multiplying
  per-unit effects along each path; this corresponds to the chain rule in a
  linear structural model, which is what `backdoor.linear_regression` fits.

Both styles return the list of `(edge, per_unit_effect)` tuples that were
applied so the caller can attribute and audit.
"""
from __future__ import annotations

from typing import Iterable

from causality_mining.graph.causal_graph import CausalGraph
from causality_mining.graph.edge import Edge


def direct_paths(graph: CausalGraph, source_id: str) -> dict[str, list[Edge]]:
    """All single-edge paths from `source_id`, indexed by target id."""
    paths: dict[str, list[Edge]] = {}
    for edge in graph.out_edges(source_id):
        paths.setdefault(edge.target, []).append(edge)
    return paths


def propagate_event(
    graph: CausalGraph,
    source_id: str,
    targets: Iterable[str] | None = None,
    max_depth: int = 1,
) -> dict[str, list[tuple[Edge, ...]]]:
    """Enumerate paths from `source_id` to each target up to `max_depth` edges.

    The returned dict maps `target_id -> list_of_paths`, where each path is a
    tuple of `Edge`s in source -> target order. `max_depth=1` is the sensible
    default for the per-edge linear-regression estimator.
    """
    target_set = set(targets) if targets is not None else {n.id for n in graph.targets()}
    out: dict[str, list[tuple[Edge, ...]]] = {tid: [] for tid in target_set}

    def walk(current: str, path: tuple[Edge, ...]) -> None:
        if len(path) >= max_depth + 1:
            return
        for edge in graph.out_edges(current):
            new_path = path + (edge,)
            if edge.target in out:
                out[edge.target].append(new_path)
            if len(new_path) < max_depth:
                walk(edge.target, new_path)

    walk(source_id, ())
    return {tid: paths for tid, paths in out.items() if paths}
