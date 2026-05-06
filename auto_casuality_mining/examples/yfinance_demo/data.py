"""Pull Mag7 OHLCV + VIX from yfinance and build a TimeSeriesCollection.

Three input timeseries kinds are exercised:

  - prices  : VECTOR  (one column per Mag7 ticker, daily close)
  - volumes : VECTOR  (one column per Mag7 ticker, daily volume)
  - vix     : SCALAR  (^VIX daily close)

News is built separately in `news.py` (LLM change report -> embedding) and
joined into the collection by `run.py`. CTR is built in `ctr.py`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm
import yfinance as yf

from causality_mining import TimeSeries, TimeSeriesCollection, TimeSeriesKind

MAG7: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA")


def _download_ohlcv(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for `tickers`. Returns a multi-indexed DataFrame."""
    return yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )


def _close_panel(data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    closes = pd.concat(
        {t: data[t]["Close"] for t in tickers if t in data.columns.get_level_values(0)},
        axis=1,
    )
    closes.columns = list(closes.columns)
    return closes.dropna(how="all")


def _volume_panel(data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    vols = pd.concat(
        {t: data[t]["Volume"] for t in tickers if t in data.columns.get_level_values(0)},
        axis=1,
    )
    vols.columns = list(vols.columns)
    return vols.dropna(how="all")


def build_market_collection(
    start: str = "2023-01-01",
    end: str | None = None,
    tickers: tuple[str, ...] = MAG7,
    cache_dir: Path | None = None,
) -> tuple[TimeSeriesCollection, pd.DataFrame, pd.DataFrame]:
    """Return `(collection, closes, volumes)`.

    `closes` and `volumes` are returned alongside the collection so the
    downstream CTR builder can derive returns and volume growth from them
    without re-downloading.
    """
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    cache_path = _market_cache_path(cache_dir, start, end, tickers)
    if cache_path and cache_path.exists() and not _refresh_cache():
        print(f"      using cached market data: {cache_path}")
        cached = pd.read_pickle(cache_path)
        closes = cached["closes"]
        volumes = cached["volumes"]
        vix = cached["vix"]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            price_future = pool.submit(_download_ohlcv, tickers, start, end)
            vix_future = pool.submit(_download_vix, start, end)
            with tqdm(total=2, desc="market: yfinance downloads", unit="req") as pbar:
                raw = price_future.result()
                pbar.update(1)
                vix_raw = vix_future.result()
                pbar.update(1)

        closes = _close_panel(raw, tickers)
        volumes = _volume_panel(raw, tickers)
        vix = vix_raw["Close"].squeeze().rename("vix").dropna()

        common_index = closes.index.intersection(volumes.index).intersection(vix.index)
        closes = closes.loc[common_index]
        volumes = volumes.loc[common_index].astype(float)
        vix = vix.loc[common_index].astype(float)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle({"closes": closes, "volumes": volumes, "vix": vix}, cache_path)

    coll = TimeSeriesCollection.from_iterable([
        TimeSeries(id="prices", kind=TimeSeriesKind.VECTOR, data=closes.astype(float)),
        TimeSeries(id="volumes", kind=TimeSeriesKind.VECTOR, data=volumes),
        TimeSeries(id="vix", kind=TimeSeriesKind.SCALAR, data=vix),
    ])
    return coll, closes, volumes


def _download_vix(start: str, end: str) -> pd.DataFrame:
    return yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)


def _market_cache_path(
    cache_dir: Path | None,
    start: str,
    end: str,
    tickers: tuple[str, ...],
) -> Path | None:
    if cache_dir is None:
        return None
    payload = {"start": start, "end": end, "tickers": list(tickers)}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / "market" / f"{key}.pkl"


def _refresh_cache() -> bool:
    return os.environ.get("YF_DEMO_REFRESH_CACHE") == "1"
