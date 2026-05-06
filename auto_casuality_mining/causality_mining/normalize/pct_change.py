"""Percentage-change normalization.

For SCALAR / VECTOR series, replace the level `x_t` with `x_t / x_{t-1} - 1`.
This is the canonical scale-invariant change measure for positive economic
quantities such as price and volume. Infinities (which arise when a previous
value is zero) are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class PctChange:
    """Replace SCALAR / VECTOR series with their percent change.

    Categorical series are passed through unchanged.
    """

    name: str = "pct_change"

    def apply(self, ts: TimeSeries) -> TimeSeries:
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return ts
        pct = ts.data.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        return TimeSeries(id=f"{ts.id}__pct_change", kind=ts.kind, data=pct)
