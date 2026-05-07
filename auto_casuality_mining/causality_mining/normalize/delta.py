"""Arithmetic-difference change encoder.

Records change as `x[b] - x[a]`, the simplest additive change measure. Useful
when the underlying signal is on a meaningful additive scale (e.g. counts, or
already-stationary deviations).

For CATEGORICAL inputs, change reduces to a one-hot indicator over the
`(prev_label, curr_label)` pair (delegated to `transition`).
"""
from __future__ import annotations

from dataclasses import dataclass

from causality_mining.normalize.transition import categorical_backward, categorical_forward
from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class Delta:
    """`Change` encoder using arithmetic differences."""

    name: str = "delta"

    def backward(self, ts: TimeSeries, horizon: int = 1) -> TimeSeries:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return categorical_backward(ts, horizon, suffix=f"{self.name}_back{horizon}")
        diffed = (ts.data - ts.data.shift(horizon)).dropna(how="all")
        return TimeSeries(id=f"{ts.id}__{self.name}_back{horizon}", kind=ts.kind, data=diffed)

    def forward(self, ts: TimeSeries, horizon: int) -> TimeSeries:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return categorical_forward(ts, horizon, suffix=f"{self.name}_fwd{horizon}")
        diffed = (ts.data.shift(-horizon) - ts.data).dropna(how="all")
        return TimeSeries(id=f"{ts.id}__{self.name}_fwd{horizon}", kind=ts.kind, data=diffed)

    def apply(self, baseline: float, delta: float) -> float:
        return float(baseline) + float(delta)
