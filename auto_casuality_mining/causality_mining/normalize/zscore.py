"""Rolling z-score normalization.

Puts every scalar/vector feature on a comparable scale so that downstream
feature-importance is not dominated by raw unit magnitudes. Uses a trailing
rolling window (`min_periods=window // 4`) so it is well-defined causally.
"""
from __future__ import annotations

from dataclasses import dataclass

from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


@dataclass(frozen=True)
class ZScore:
    """Trailing rolling z-score for SCALAR and VECTOR series."""

    window: int = 64
    name: str = "zscore"

    def apply(self, ts: TimeSeries) -> TimeSeries:
        if ts.kind is TimeSeriesKind.CATEGORICAL:
            return ts
        min_periods = max(2, self.window // 4)
        rolled = ts.data.rolling(self.window, min_periods=min_periods)
        mean = rolled.mean()
        std = rolled.std().replace(0.0, float("nan"))
        zs = (ts.data - mean) / std
        return TimeSeries(id=f"{ts.id}__z", kind=ts.kind, data=zs.dropna(how="all"))
