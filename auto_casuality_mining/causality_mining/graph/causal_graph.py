"""`CausalGraph`: the container produced by curation and consumed by inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from causality_mining.graph.edge import Edge
from causality_mining.graph.node import Node


@dataclass
class CausalGraph:
    """Directed graph of `Node`s and `Edge`s, with simple lookup helpers.

    The graph is intentionally lightweight: it carries enough metadata for the
    inference engine to (a) walk source -> target paths and (b) hand each edge
    to DoWhy as a single-treatment / single-outcome causal model.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str, int], Edge] = field(default_factory=dict)

    @classmethod
    def from_components(cls, nodes: Iterable[Node], edges: Iterable[Edge]) -> "CausalGraph":
        g = cls()
        for n in nodes:
            g.add_node(n)
        for e in edges:
            g.add_edge(e)
        return g

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes:
            raise KeyError(f"unknown source node: {edge.source}")
        if edge.target not in self.nodes:
            raise KeyError(f"unknown target node: {edge.target}")
        self.edges[edge.key()] = edge

    def out_edges(self, source: str) -> list[Edge]:
        return [e for e in self.edges.values() if e.source == source]

    def in_edges(self, target: str) -> list[Edge]:
        return [e for e in self.edges.values() if e.target == target]

    def targets(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_target]

    def __iter__(self) -> Iterator[Edge]:
        return iter(self.edges.values())

    def __len__(self) -> int:
        return len(self.edges)

    def to_json(self, path: str) -> None:
        from causality_mining.graph.io import save_json

        save_json(self, path)

    @classmethod
    def from_json(cls, path: str) -> "CausalGraph":
        from causality_mining.graph.io import load_json

        return load_json(path)
