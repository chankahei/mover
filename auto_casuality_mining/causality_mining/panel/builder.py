"""Glue: a `TimeSeriesCollection` -> a single (T x F) DataFrame plus column layout.

The layout records, for each output column, which `TimeSeries.id` it came from.
Curation needs this map to translate per-column SHAP importances back into
per-series candidate edges (a vector series can produce many columns but maps
to a single graph node).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from causality_mining.panel.resample import resample_to_grid
from causality_mining.timeseries.collection import TimeSeriesCollection


@dataclass(frozen=True)
class ColumnLayout:
    """Mapping from panel column name -> originating timeseries id."""

    column_to_series: dict[str, str]

    def series_columns(self, series_id: str) -> list[str]:
        return [c for c, sid in self.column_to_series.items() if sid == series_id]


def build_panel(collection: TimeSeriesCollection, freq: str) -> tuple[pd.DataFrame, ColumnLayout]:
    """Resample every series to `freq` and concatenate into one DataFrame.

    Vector series contribute multiple columns but all map back to the single
    series id in the returned `ColumnLayout`.
    """
    frames: list[pd.DataFrame] = []
    column_to_series: dict[str, str] = {}
    for ts in collection:
        df = resample_to_grid(ts, freq)
        for col in df.columns:
            column_to_series[col] = ts.id
        frames.append(df)
    if not frames:
        return pd.DataFrame(), ColumnLayout({})
    panel = pd.concat(frames, axis=1).sort_index()
    return panel, ColumnLayout(column_to_series)
