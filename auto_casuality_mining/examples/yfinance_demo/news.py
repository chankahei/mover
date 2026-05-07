"""Build a news VECTOR timeseries via LLM change reports + embeddings.

Per-day pipeline:

    headlines_today  ─┐
                      ├──> LLMClient.chat ──> change_report ──> EmbeddingClient.embed ──> v_t
    headlines_yesterday─┘

The per-day vectors are emitted as-is (full embedding dim) as a VECTOR
`TimeSeries` with `pre_normalized=True` (the report already encodes change
semantics, so the package's `Change` encoder should NOT touch it).
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import yfinance as yf

from causality_mining import TimeSeries, TimeSeriesKind

from embed import EmbeddingClient
from llm import LLMClient

_SYSTEM_PROMPT = (
    "You are a financial-news delta extractor. Compare ONLY today's headlines against "
    "yesterday's headlines and output only what is materially new or materially changed. "
    "Material means: a new theme, a clear escalation/de-escalation, a reversal, or a new "
    "company-specific risk/opportunity likely to impact sentiment. Ignore repeated or merely "
    "rephrased headlines. Do not summarize both days. Do not add background facts. "
    "Output format must be exact: if no material change exists, output exactly "
    "'NO_MATERIAL_CHANGE'. Otherwise output 1-5 lines, each line starting with '- ' and "
    "containing one concrete change statement."
)


def _fetch_headlines_by_day(tickers: Iterable[str], cache_dir: Path) -> dict[date, list[str]]:
    """Aggregate yfinance news across tickers into `{day: [headline, ...]}`."""
    ticker_list = list(tickers)
    if not ticker_list:
        return {}
    cache_path = _headlines_cache_path(cache_dir, ticker_list)
    if cache_path.exists() and not _refresh_cache():
        print(f"      using cached news headlines: {cache_path}")
        return _load_headlines_cache(cache_path)

    by_day: dict[date, list[str]] = defaultdict(list)

    workers = _news_workers(default=8)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one_ticker_news, ticker): ticker for ticker in ticker_list}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="news: fetching ticker headlines",
            unit="ticker",
        ):
            for day, headline in fut.result():
                by_day[day].append(headline)
    out = dict(sorted(by_day.items()))
    _save_headlines_cache(cache_path, out)
    return out


def _parse_iso(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def _yesterday_for(day: date, sorted_days: list[date], by_day: dict[date, list[str]]) -> list[str]:
    """Most recent prior day with at least one headline."""
    for d in reversed(sorted_days):
        if d < day and by_day[d]:
            return by_day[d]
    return []


def _fetch_one_ticker_news(ticker: str) -> list[tuple[date, str]]:
    """Fetch one ticker's headlines and return `(day, tagged_title)` rows."""
    rows: list[tuple[date, str]] = []
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return rows
    for item in items:
        content = item.get("content") or item
        ts = (
            content.get("providerPublishTime")
            or item.get("providerPublishTime")
            or _parse_iso(content.get("pubDate"))
        )
        title = content.get("title") or item.get("title")
        if ts is None or not title:
            continue
        day = datetime.utcfromtimestamp(int(ts)).date()
        rows.append((day, f"[{ticker}] {title}"))
    return rows


def _news_workers(default: int) -> int:
    configured = os.environ.get("YF_DEMO_MAX_WORKERS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return default


def _headlines_cache_path(cache_dir: Path, tickers: list[str]) -> Path:
    payload = {"tickers": sorted(tickers)}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / "news" / f"headlines_{key}.json"


def _save_headlines_cache(path: Path, by_day: dict[date, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {day.isoformat(): items for day, items in by_day.items()}
    path.write_text(json.dumps(payload))


def _load_headlines_cache(path: Path) -> dict[date, list[str]]:
    payload = json.loads(path.read_text())
    out: dict[date, list[str]] = {}
    for day_text, items in payload.items():
        try:
            day = date.fromisoformat(day_text)
        except ValueError:
            continue
        out[day] = [str(item) for item in items]
    return dict(sorted(out.items()))


def _refresh_cache() -> bool:
    return os.environ.get("YF_DEMO_REFRESH_CACHE") == "1"


def build_news_timeseries(
    tickers: Iterable[str],
    market_index: pd.DatetimeIndex,
    cache_dir: Path,
) -> TimeSeries:
    """Return a daily VECTOR TimeSeries of change-report embeddings.

    Days with no news produce a zero vector so the panel index stays dense.
    """
    by_day = _fetch_headlines_by_day(tickers, cache_dir=cache_dir)
    sorted_days = sorted(by_day.keys())

    llm = LLMClient(cache_dir=cache_dir / "llm")
    embed = EmbeddingClient(cache_dir=cache_dir / "embed")

    raw_vectors: dict[pd.Timestamp, np.ndarray] = {}
    work_items: list[tuple[pd.Timestamp, str]] = []
    for ts in market_index:
        today = by_day.get(ts.date(), [])
        if not today:
            continue
        yesterday = _yesterday_for(ts.date(), sorted_days, by_day)
        user_prompt = (
            f"Yesterday:\n{chr(10).join(yesterday) or '(none)'}\n\nToday:\n{chr(10).join(today)}"
        )
        work_items.append((ts, user_prompt))

    workers = _news_workers(default=6)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_embed_one_market_day, ts, user_prompt, llm, embed): ts
            for ts, user_prompt in work_items
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="news: llm + embedding",
            unit="day",
        ):
            ts, vec = fut.result()
            raw_vectors[ts] = vec

    if raw_vectors:
        times = sorted(raw_vectors.keys())
        stacked = np.stack([raw_vectors[t] for t in times], axis=0)
    else:
        # No news landed on any market day; embed a placeholder once just to
        # discover the model's vector dim so the zero-filled series is well-shaped.
        stacked = embed.embed("no news").reshape(1, -1)
        times = []

    dim = stacked.shape[1]
    columns = [f"e{i}" for i in range(dim)]
    full_df = pd.DataFrame(0.0, index=market_index, columns=columns)
    if times:
        embedded_df = pd.DataFrame(stacked, index=pd.DatetimeIndex(times), columns=columns)
        full_df.loc[embedded_df.index] = embedded_df.values
    return TimeSeries(id="news", kind=TimeSeriesKind.VECTOR, data=full_df, pre_normalized=True)


def _embed_one_market_day(
    ts: pd.Timestamp,
    user_prompt: str,
    llm: LLMClient,
    embed: EmbeddingClient,
) -> tuple[pd.Timestamp, np.ndarray]:
    report = llm.chat(system=_SYSTEM_PROMPT, user=user_prompt)
    return ts, embed.embed(report)
