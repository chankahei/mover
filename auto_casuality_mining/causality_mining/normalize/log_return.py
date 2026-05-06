"""Log-return normalization.

For positive-valued scalar / vector series (e.g. price, volume), `log(x_t / x_{t-1})`
is the canonical scale-invariant change measure. Falls back to NaN where input
is non-positive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class LogReturn:
    """Replace strictly positive SCALAR / VECTOR series with log returns."""

    name: str = "log_return"

    def apply(self, ts: TimeSeries) -> TimeSeries:
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return ts
        positive = ts.data.where(ts.data > 0)
        log_ret = np.log(positive).diff().dropna(how="all")
        return TimeSeries(id=f"{ts.id}__log_return", kind=ts.kind, data=log_ret)
