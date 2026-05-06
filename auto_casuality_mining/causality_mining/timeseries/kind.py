"""Enumerates the three supported input timeseries kinds."""
from __future__ import annotations

from enum import Enum


class TimeSeriesKind(str, Enum):
    """Kinds of input timeseries supported by curation and inference.

    - SCALAR: 1-D real-valued (e.g. price, return, dwell time).
    - CATEGORICAL: discrete labels (e.g. article frame, market regime).
    - VECTOR: multidimensional real-valued (e.g. embedding, factor loadings).
    """

    SCALAR = "scalar"
    CATEGORICAL = "categorical"
    VECTOR = "vector"
