"""A collection of named `TimeSeries` to feed into curation / inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from causality_mining.timeseries.series import TimeSeries


@dataclass
class TimeSeriesCollection:
    """An ordered, name-keyed bag of `TimeSeries`."""

    series: dict[str, TimeSeries] = field(default_factory=dict)

    @classmethod
    def from_iterable(cls, items: Iterable[TimeSeries]) -> "TimeSeriesCollection":
        c = cls()
        for ts in items:
            c.add(ts)
        return c

    def add(self, ts: TimeSeries) -> None:
        if ts.id in self.series:
            raise ValueError(f"duplicate timeseries id: {ts.id}")
        self.series[ts.id] = ts

    def get(self, series_id: str) -> TimeSeries:
        if series_id not in self.series:
            raise KeyError(f"unknown timeseries id: {series_id}")
        return self.series[series_id]

    def ids(self) -> list[str]:
        return list(self.series.keys())

    def __iter__(self) -> Iterator[TimeSeries]:
        return iter(self.series.values())

    def __len__(self) -> int:
        return len(self.series)

    def __contains__(self, series_id: str) -> bool:
        return series_id in self.series
