"""First-difference (delta) normalization.

Records the change from the previous observation instead of the raw level.
This is one of the most universally useful normalizations for causal mining
because levels are usually non-stationary while changes are stationary.
"""
from __future__ import annotations

from dataclasses import dataclass

from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class Delta:
    """Replace SCALAR / VECTOR series with their first difference.

    Categorical series are passed through unchanged.
    """

    name: str = "delta"

    def apply(self, ts: TimeSeries) -> TimeSeries:
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return ts
        diffed = ts.data.diff().dropna(how="all")
        return TimeSeries(id=f"{ts.id}__delta", kind=ts.kind, data=diffed)
