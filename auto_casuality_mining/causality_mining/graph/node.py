"""Causal graph node: a 1:1 reference to an input `TimeSeries`."""
from __future__ import annotations

from dataclasses import dataclass

from causality_mining.timeseries.kind import TimeSeriesKind


@dataclass(frozen=True)
class Node:
    """A node in the causal graph.

    Each node corresponds to one named timeseries. For VECTOR series, the node
    stands in for the whole multidimensional bundle; the panel layout knows how
    to expand it into per-column features when an estimator needs them.

    Attributes:
        id: stable identifier (matches `TimeSeries.id`).
        kind: kind of the originating timeseries.
        is_target: True if this node is a prediction target (CTR, trade, ...).
    """

    id: str
    kind: TimeSeriesKind
    is_target: bool = False
