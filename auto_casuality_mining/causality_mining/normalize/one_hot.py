"""One-hot expansion of categorical timeseries.

Causal discovery and DoWhy estimators are easier to apply when categorical
levels are represented as scalar indicators. This transform replaces a single
CATEGORICAL series with a VECTOR series whose columns are the levels.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class OneHot:
    """Expand CATEGORICAL series into a VECTOR series of 0/1 indicators."""

    name: str = "one_hot"

    def apply(self, ts: TimeSeries) -> TimeSeries:
        if ts.kind is not TimeSeriesKind.CATEGORICAL:
            return ts
        dummies = pd.get_dummies(ts.data, dtype=float)
        dummies.columns = [str(c) for c in dummies.columns]
        return TimeSeries(id=f"{ts.id}__oh", kind=TimeSeriesKind.VECTOR, data=dummies)
