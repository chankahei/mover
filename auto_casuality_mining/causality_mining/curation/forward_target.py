"""Build a forward-change target column on the panel grid.

Used by both the `discover_graph` and `curate_graph` flows. Centralizes the
two-step pipeline:

    1. `encode_forward_target(raw_ts, change, lag)` -> a `TimeSeries` whose
       value at time t is the change of the original series from t to t+lag.
    2. `resample_to_grid(fwd_ts, freq)` -> a single-column DataFrame on the
       same grid as the feature panel.

Returns the resulting Series (single column). Raises if the encoded target is
multi-column (e.g. forward change of a CATEGORICAL produces a VECTOR), since
the downstream estimators only fit single-output targets.
"""
from __future__ import annotations

import pandas as pd

from causality_mining.normalize.base import Change
from causality_mining.normalize.encode import encode_forward_target
from causality_mining.panel.resample import resample_to_grid
from causality_mining.timeseries.series import TimeSeries


def forward_target_column(
    raw_target_ts: TimeSeries,
    change: Change,
    freq: str,
    lag: int,
) -> pd.Series:
    """Forward `lag`-step change of `raw_target_ts`, resampled to `freq`."""
    fwd_ts = encode_forward_target(raw_target_ts, change, lag)
    df = resample_to_grid(fwd_ts, freq)
    if df.shape[1] != 1:
        raise ValueError(
            f"forward target for {raw_target_ts.id} produced {df.shape[1]} columns; "
            f"expected exactly 1 (kind={fwd_ts.kind.value}). "
            f"Targets that encode forward change as a vector "
            f"(e.g. CATEGORICAL transition encoding) are not supported here."
        )
    return df.iloc[:, 0]
