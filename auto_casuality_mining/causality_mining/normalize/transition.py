"""One-hot transition encoding for categorical change.

The change of a categorical series between two timestamps is encoded as a
one-hot indicator over the observed `(prev_label, curr_label)` pair. The
output is a VECTOR series whose columns enumerate all transitions actually
observed in the input window.

Used by every `Change` encoder for CATEGORICAL inputs (so the encoder choice
only matters for numeric series).
"""
from __future__ import annotations

import pandas as pd

from causality_mining.timeseries.kind import TimeSeriesKind
from causality_mining.timeseries.series import TimeSeries


_PAIR_SEP = "->"


def _encode_pairs(prev: pd.Series, curr: pd.Series, prefix: str) -> pd.DataFrame:
    """Build a one-hot DataFrame over `(prev, curr)` label pairs.

    Rows where either side is NaN are dropped from the output. Column names are
    `f"{prefix}__{prev}->{curr}"` so callers can trace which transition each
    indicator corresponds to.
    """
    valid = prev.notna() & curr.notna()
    pairs_str = prev.astype("string").str.cat(curr.astype("string"), sep=_PAIR_SEP)
    pairs = pairs_str.where(valid)
    dummies = pd.get_dummies(pairs, dummy_na=False, dtype=float)
    dummies.columns = [f"{prefix}__{c}" for c in dummies.columns]
    return dummies.dropna(how="all")


def categorical_backward(ts: TimeSeries, horizon: int, suffix: str) -> TimeSeries:
    """Backward `horizon`-step transition encoding of a CATEGORICAL series."""
    if ts.kind is not TimeSeriesKind.CATEGORICAL:
        raise TypeError(f"{ts.id}: categorical_backward requires CATEGORICAL input")
    prev = ts.data.shift(horizon)
    dummies = _encode_pairs(prev, ts.data, prefix="t")
    return TimeSeries(id=f"{ts.id}__{suffix}", kind=TimeSeriesKind.VECTOR, data=dummies)


def categorical_forward(ts: TimeSeries, horizon: int, suffix: str) -> TimeSeries:
    """Forward `horizon`-step transition encoding of a CATEGORICAL series."""
    if ts.kind is not TimeSeriesKind.CATEGORICAL:
        raise TypeError(f"{ts.id}: categorical_forward requires CATEGORICAL input")
    nxt = ts.data.shift(-horizon)
    dummies = _encode_pairs(ts.data, nxt, prefix="t")
    return TimeSeries(id=f"{ts.id}__{suffix}", kind=TimeSeriesKind.VECTOR, data=dummies)
