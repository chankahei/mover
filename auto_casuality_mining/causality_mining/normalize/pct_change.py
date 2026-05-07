"""Percentage-change encoder for positive scalar / vector signals.

Records change as `x[b] / x[a] - 1`, the canonical scale-invariant measure for
strictly-positive economic quantities (price, volume). Multi-step pct change is
computed against the actual endpoints, NOT by summing single-step pct changes
(which would be wrong because pct change is multiplicative, not additive).

For CATEGORICAL inputs, change reduces to a one-hot indicator over the
`(prev_label, curr_label)` pair (delegated to `transition`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causality_mining.normalize.transition import categorical_backward, categorical_forward
from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


def _ratio_change(numerator, denominator):
    return (numerator / denominator - 1).replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class PctChange:
    """`Change` encoder using percentage difference (`x[b] / x[a] - 1`)."""

    name: str = "pct_change"

    def backward(self, ts: TimeSeries, horizon: int = 1) -> TimeSeries:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return categorical_backward(ts, horizon, suffix=f"{self.name}_back{horizon}")
        pct = _ratio_change(ts.data, ts.data.shift(horizon)).dropna(how="all")
        return TimeSeries(id=f"{ts.id}__{self.name}_back{horizon}", kind=ts.kind, data=pct)

    def forward(self, ts: TimeSeries, horizon: int) -> TimeSeries:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return categorical_forward(ts, horizon, suffix=f"{self.name}_fwd{horizon}")
        pct = _ratio_change(ts.data.shift(-horizon), ts.data).dropna(how="all")
        return TimeSeries(id=f"{ts.id}__{self.name}_fwd{horizon}", kind=ts.kind, data=pct)

    def apply(self, baseline: float, delta: float) -> float:
        return float(baseline) * (1.0 + float(delta))
