"""Apply a `Change` encoder to a collection or to a single series.

These helpers centralize the `pre_normalized` rule:

- `encode_features`: builds a backward-change feature collection from a
  heterogeneous input collection. Series flagged `pre_normalized=True` pass
  through unchanged (they already encode change semantics).
- `encode_forward_target`: builds a forward-change target series at a given
  horizon. For `pre_normalized=True` series the forward "change" reduces to a
  pure time shift, since we cannot recover the underlying raw level.
"""
from __future__ import annotations

from causality_mining.normalize.base import Change
from causality_mining.timeseries.collection import TimeSeriesCollection
from causality_mining.timeseries.series import TimeSeries


def encode_features(collection: TimeSeriesCollection, change: Change) -> TimeSeriesCollection:
    """Backward 1-step change of every non-`pre_normalized` series in `collection`."""
    out = TimeSeriesCollection()
    for ts in collection:
        out.add(ts if ts.pre_normalized else change.backward(ts, horizon=1))
    return out


def encode_forward_target(ts: TimeSeries, change: Change, horizon: int) -> TimeSeries:
    """Forward `horizon`-step change of a single target series.

    `pre_normalized=True` short-circuits to `ts.data.shift(-horizon)` because
    the underlying raw level is no longer recoverable.
    """
    if not ts.pre_normalized:
        return change.forward(ts, horizon)
    shifted = ts.data.shift(-horizon).dropna(how="all")
    return TimeSeries(id=f"{ts.id}__fwd{horizon}", kind=ts.kind, data=shifted)
