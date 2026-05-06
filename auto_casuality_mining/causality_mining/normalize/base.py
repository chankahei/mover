"""Normalizer protocol.

A `Normalizer` takes a `TimeSeries` and returns a (possibly different) one.
Most normalizers are kind-specific and pass through series of other kinds
unchanged so a global `Pipeline` can be applied to a heterogeneous collection.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from causality_mining.timeseries.series import TimeSeries


@runtime_checkable
class Normalizer(Protocol):
    """Stateless transform from `TimeSeries` to `TimeSeries`."""

    name: str

    def apply(self, ts: TimeSeries) -> TimeSeries:  # pragma: no cover - protocol
        ...
