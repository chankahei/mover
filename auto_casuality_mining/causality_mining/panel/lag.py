"""Build lagged copies of feature columns.

Causal-graph curation predicts target_t from feature_{t-l}; this module
produces those `feature__lag<l>` columns. Lagging at the panel layer (not the
estimator layer) is what enforces temporally plausible candidate edges and
prevents same-timestamp leakage.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def lagged_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    lags: Iterable[int],
) -> pd.DataFrame:
    """Return a DataFrame that holds, for each (column, lag) pair, a shifted copy.

    The output column name is `f"{col}__lag{lag}"`. Lag 0 is included only if
    explicitly requested; the typical curation use case asks for lags >= 1.
    """
    out: dict[str, pd.Series] = {}
    cols = list(columns)
    for lag in lags:
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        for col in cols:
            out[f"{col}__lag{lag}"] = df[col].shift(lag)
    return pd.DataFrame(out, index=df.index)
