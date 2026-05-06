"""A new event on one of the input timeseries.

The architecture doc's "Event Flow" begins with a single signal hitting the
event bus. From the causality engine's point of view, that's just one new
observation on a timeseries node we already have in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class NewEvent:
    """One fresh observation on a known node.

    Attributes:
        series_id: id of the source timeseries (must be a node in the graph).
        timestamp: when the event occurred.
        value: observed value.
            - SCALAR: a float.
            - CATEGORICAL: a level (string).
            - VECTOR: a mapping {column_name -> float}.
    """

    series_id: str
    timestamp: pd.Timestamp
    value: Any
