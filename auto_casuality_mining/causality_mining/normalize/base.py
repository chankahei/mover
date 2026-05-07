"""Change-encoding protocol used by causal discovery and inference.

A `Change` knows how to compute the *change* of a `TimeSeries` over a horizon,
in BOTH temporal directions:

- `backward(ts, horizon=h)`: at time t, the change observed from t-h to t.
  Used to build SOURCE features ("the 1-period change just observed").

- `forward(ts, horizon=h)`: at time t, the change that will occur from t to t+h.
  Used to build TARGET signals ("the h-period change about to unfold").

Why both, instead of summing 1-period backward changes? Because change is not
generally additive: pct-change does not sum, log-return does (log p_{t+h}/p_t
== sum of log returns), and arithmetic delta does. Each encoder defines its own
arithmetic so multi-step change is computed correctly per its semantics.

Categorical handling is shared across encoders: a categorical "change" between
two timestamps is the one-hot indicator of the (old_label, new_label) pair, so
every `Change` returns a VECTOR series for CATEGORICAL inputs regardless of
encoder.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from causality_mining.timeseries.series import TimeSeries


@runtime_checkable
class Change(Protocol):
    """Stateless change encoder for time series."""

    name: str

    def backward(self, ts: TimeSeries, horizon: int = 1) -> TimeSeries:  # pragma: no cover - protocol
        """Backward `horizon`-step change of `ts` aligned at index t."""
        ...

    def forward(self, ts: TimeSeries, horizon: int) -> TimeSeries:  # pragma: no cover - protocol
        """Forward `horizon`-step change of `ts` aligned at index t."""
        ...

    def apply(self, baseline: float, delta: float) -> float:  # pragma: no cover - protocol
        """Combine a raw `baseline` level with a `delta` in this encoder's units.

        Inverse of `forward` at the level scale: if `delta == forward(ts, h)[t]`
        and `baseline == ts.data[t]` (numeric), then `apply(baseline, delta)`
        equals `ts.data[t+h]`.
        """
        ...
