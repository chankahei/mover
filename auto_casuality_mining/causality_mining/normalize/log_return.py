"""Log-return change encoder for strictly-positive signals.

Records change as `log(x[b]) - log(x[a])`. Log returns are additive across time
(log(x_{t+h}/x_t) == sum of single-step log returns), but this encoder still
computes multi-step change directly against the endpoints to keep the API
parallel with `Delta` and `PctChange`.

For CATEGORICAL inputs, change reduces to a one-hot indicator over the
`(prev_label, curr_label)` pair (delegated to `transition`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causality_mining.normalize.transition import categorical_backward, categorical_forward
from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class LogReturn:
    """`Change` encoder using log differences (`log(x[b]) - log(x[a])`)."""

    name: str = "log_return"

    def backward(self, ts: TimeSeries, horizon: int = 1) -> TimeSeries:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return categorical_backward(ts, horizon, suffix=f"{self.name}_back{horizon}")
        positive = ts.data.where(ts.data > 0)
        log_ret = (np.log(positive) - np.log(positive.shift(horizon))).dropna(how="all")
        return TimeSeries(id=f"{ts.id}__{self.name}_back{horizon}", kind=ts.kind, data=log_ret)

    def forward(self, ts: TimeSeries, horizon: int) -> TimeSeries:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return categorical_forward(ts, horizon, suffix=f"{self.name}_fwd{horizon}")
        positive = ts.data.where(ts.data > 0)
        log_ret = (np.log(positive.shift(-horizon)) - np.log(positive)).dropna(how="all")
        return TimeSeries(id=f"{ts.id}__{self.name}_fwd{horizon}", kind=ts.kind, data=log_ret)

    def apply(self, baseline: float, delta: float) -> float:
        return float(baseline) * float(np.exp(delta))
