"""Utilities to explode vector series into per-column scalar nodes."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from causality_mining import TimeSeries, TimeSeriesCollection, TimeSeriesKind


def explode_vector_series(
    collection: TimeSeriesCollection,
    series_ids: Iterable[str],
) -> TimeSeriesCollection:
    """Return a new collection with selected VECTOR series exploded by column.

    For each selected VECTOR series `prices` with columns `{AAPL, MSFT, ...}`,
    this creates SCALAR series `{prices_AAPL, prices_MSFT, ...}` and drops the
    original vector node from the returned collection.
    """
    selected = set(series_ids)
    exploded: list[TimeSeries] = []
    for ts in collection:
        if ts.id not in selected or ts.kind is not TimeSeriesKind.VECTOR:
            exploded.append(ts)
            continue
        for col in ts.data.columns:
            series_id = f"{ts.id}_{col}"
            series = pd.Series(ts.data[col].astype(float), index=ts.data.index, name=series_id)
            exploded.append(
                TimeSeries(
                    id=series_id,
                    kind=TimeSeriesKind.SCALAR,
                    data=series,
                    pre_normalized=ts.pre_normalized,
                )
            )
    return TimeSeriesCollection.from_iterable(exploded)
