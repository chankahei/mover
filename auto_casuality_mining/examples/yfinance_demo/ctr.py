"""Synthesize fake per-ticker CTR targets from same-day market changes.

For each ticker `t`:

    r_{t,d} = close_{t,d} / close_{t,d-1} - 1
    g_{t,d} = volume_{t,d} / volume_{t,d-1} - 1
    ctr_{t,d} = 0.6 * |r_{t,d}| + 0.4 * clip(g_{t,d}, -1, 1) + noise

So each ticker's CTR on day `d` is generated from the SAME day's price and
volume movement for that same ticker. Each emitted `TimeSeries` is marked
`pre_normalized=True` so the discovery `PctChange` pipeline does not divide
CTR by its previous value (which is often near zero). The pipeline only
percent-changes price/volume, and discovery should recover

    prices_<t>__pct_change __lag0  ->  ctr_<t>      (non-linear via |.|)
    volumes_<t>__pct_change __lag0 ->  ctr_<t>      (linear)

with LightGBM + Tree SHAP picking up the non-linear absolute-return effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from causality_mining import TimeSeries, TimeSeriesKind


def build_ctr(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    noise_std: float = 0.005,
    seed: int = 0,
) -> list[TimeSeries]:
    """Produce one daily SCALAR CTR series per ticker (pre-normalized)."""
    returns = closes.pct_change()
    vol_growth = volumes.pct_change().clip(-1.0, 1.0)
    rng = np.random.default_rng(seed)
    out: list[TimeSeries] = []
    for ticker in closes.columns:
        raw = 0.6 * returns[ticker].abs() + 0.4 * vol_growth[ticker]
        ctr = raw + pd.Series(rng.normal(0, noise_std, len(raw)), index=raw.index)
        ctr = ctr.fillna(0.0).clip(lower=0.0).rename(f"ctr_{ticker}")
        out.append(
            TimeSeries(
                id=f"ctr_{ticker}",
                kind=TimeSeriesKind.SCALAR,
                data=ctr,
                pre_normalized=True,
            )
        )
    return out
