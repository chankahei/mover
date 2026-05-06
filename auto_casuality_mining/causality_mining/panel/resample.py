"""Resample heterogeneous timeseries onto a single common time grid."""
from __future__ import annotations

import pandas as pd

from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


def resample_to_grid(ts: TimeSeries, freq: str) -> pd.DataFrame:
    """Resample a `TimeSeries` to a fixed pandas frequency string.

    - SCALAR / VECTOR: aggregated by mean.
    - CATEGORICAL: aggregated by last observation, then forward-filled.

    Returns a 2-D DataFrame so downstream code can treat all kinds uniformly.
    """
    if ts.kind is TimeSeriesKind.CATEGORICAL:
        resampled = ts.data.resample(freq).last().ffill()
        return resampled.to_frame(name=ts.id)
    if ts.kind is TimeSeriesKind.SCALAR:
        return ts.data.resample(freq).mean().to_frame(name=ts.id)
    df = ts.data.resample(freq).mean()
    df.columns = [f"{ts.id}__{c}" for c in df.columns]
    return df
