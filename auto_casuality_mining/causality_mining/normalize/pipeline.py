"""Compose normalizers and apply them across a `TimeSeriesCollection`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from causality_mining.normalize.base import Normalizer
from causality_mining.timeseries.collection import TimeSeriesCollection
from causality_mining.timeseries.series import TimeSeries


@dataclass
class Pipeline:
    """Apply a sequence of `Normalizer`s, in order, to every series."""

    steps: Sequence[Normalizer] = field(default_factory=tuple)

    def apply(self, ts: TimeSeries) -> TimeSeries:
        if ts.pre_normalized:
            return ts
        for step in self.steps:
            ts = step.apply(ts)
        return ts

    def apply_collection(self, collection: TimeSeriesCollection) -> TimeSeriesCollection:
        out = TimeSeriesCollection()
        for ts in collection:
            transformed = self.apply(ts)
            out.add(transformed)
        return out
