"""Single typed timeseries object.

A `TimeSeries` wraps either a 1-D pandas Series (SCALAR / CATEGORICAL) or a
2-D pandas DataFrame (VECTOR). The index is always a `pd.DatetimeIndex`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import pandas as pd

from causality_mining.timeseries.kind import TimeSeriesKind

Frame = Union[pd.Series, pd.DataFrame]


@dataclass(frozen=True)
class TimeSeries:
    """A typed, named timeseries with a `DatetimeIndex`.

    Attributes:
        id: stable identifier, used as a node id in the causal graph.
        kind: SCALAR, CATEGORICAL, or VECTOR.
        data: pd.Series for scalar/categorical, pd.DataFrame for vector.
        pre_normalized: if True, every `Normalizer` in a `Pipeline` will
            pass this series through unchanged. Use this when the values
            already encode "change" semantics (e.g. an LLM-generated
            day-over-day report embedding) so the default `Delta` /
            `LogReturn` / `ZScore` transforms do not double-difference it.
    """

    id: str
    kind: TimeSeriesKind
    data: Frame
    pre_normalized: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError(f"{self.id}: data must be indexed by DatetimeIndex")
        if self.kind in (TimeSeriesKind.SCALAR, TimeSeriesKind.CATEGORICAL):
            if not isinstance(self.data, pd.Series):
                raise TypeError(f"{self.id}: {self.kind.value} requires pd.Series")
        elif self.kind is TimeSeriesKind.VECTOR:
            if not isinstance(self.data, pd.DataFrame):
                raise TypeError(f"{self.id}: vector requires pd.DataFrame")
            if self.data.shape[1] < 1:
                raise ValueError(f"{self.id}: vector must have >= 1 column")

    @property
    def width(self) -> int:
        """Number of feature columns this series contributes to a panel."""
        if self.kind is TimeSeriesKind.VECTOR:
            return int(self.data.shape[1])
        return 1

    def column_names(self) -> list[str]:
        """Panel column names this series will produce (before lagging)."""
        if self.kind is TimeSeriesKind.VECTOR:
            return [f"{self.id}__{c}" for c in self.data.columns]
        return [self.id]
