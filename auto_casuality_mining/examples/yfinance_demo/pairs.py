"""For every edge in the curated graph, render a timeseries pair plot.

The plot overlays:

  - The source series's backward 1-step change at time t (left axis).
  - The target series's forward `lag`-step change between t and t+lag (right axis).

So a human can eyeball whether the discovered relationship "1-step change in
source at t -> lag-step change in target between t and t+lag" looks plausible.

For VECTOR sources, the cross-column mean is plotted (the curation pipeline
already collapsed each VECTOR series into a single panel column internally).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from causality_mining import CausalGraph, TimeSeriesCollection, TimeSeriesKind
from causality_mining.normalize.base import Change
from causality_mining.normalize.encode import encode_forward_target


def _series_as_line(coll: TimeSeriesCollection, series_id: str) -> pd.Series:
    """Reduce any series kind to a single 1-D pd.Series for plotting."""
    ts = coll.get(series_id)
    if ts.kind is TimeSeriesKind.VECTOR:
        return ts.data.mean(axis=1).rename(series_id)
    if ts.kind is TimeSeriesKind.CATEGORICAL:
        codes, _ = pd.factorize(ts.data)
        return pd.Series(codes, index=ts.data.index, name=series_id, dtype=float)
    return ts.data.rename(series_id)


def _forward_target_line(
    coll: TimeSeriesCollection,
    series_id: str,
    change: Change,
    lag: int,
) -> pd.Series:
    """Forward `lag`-step change of `series_id`, collapsed to a 1-D Series."""
    ts = coll.get(series_id)
    fwd_ts = encode_forward_target(ts, change, lag)
    if isinstance(fwd_ts.data, pd.DataFrame):
        return fwd_ts.data.mean(axis=1).rename(series_id)
    return fwd_ts.data.rename(series_id)


def render_edge_pairs(
    graph: CausalGraph,
    feature_collection: TimeSeriesCollection,
    raw_collection: TimeSeriesCollection,
    change: Change,
    out_dir: Path,
) -> list[Path]:
    """Write one PNG per promoted edge. Returns the list of paths written.

    `feature_collection` holds backward 1-step change series (source side).
    `raw_collection` holds the un-encoded series so we can compute forward
    `lag`-step change of the target (target side).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded_to_raw = {
        encoded.id: raw.id for raw, encoded in zip(raw_collection, feature_collection)
    }
    paths: list[Path] = []
    for edge in tqdm(
        graph.edges.values(),
        total=len(graph.edges),
        desc="viz: edge pair plots",
        unit="edge",
    ):
        src = _series_as_line(feature_collection, edge.source)
        raw_target_id = encoded_to_raw.get(edge.target, edge.target)
        tgt = _forward_target_line(raw_collection, raw_target_id, change, edge.lag)

        fig, ax_left = plt.subplots(figsize=(10, 4))
        ax_right = ax_left.twinx()
        ax_left.plot(src.index, src.values, color="#1f77b4", label=f"{edge.source} (back 1)")
        ax_right.plot(tgt.index, tgt.values, color="#d35400", label=f"{edge.target} (fwd {edge.lag})")

        ax_left.set_ylabel(f"{edge.source} (back 1)", color="#1f77b4")
        ax_right.set_ylabel(f"{edge.target} (fwd {edge.lag})", color="#d35400")
        ax_left.tick_params(axis="y", labelcolor="#1f77b4")
        ax_right.tick_params(axis="y", labelcolor="#d35400")

        title = (
            f"{edge.source} -> {edge.target}    "
            f"lag={edge.lag}  strength={edge.strength:+.4f}  "
            f"conf={edge.confidence:.2f}  imp={edge.importance:.3f}"
        )
        ax_left.set_title(title)
        fig.tight_layout()

        out_path = out_dir / f"{edge.source}__{edge.target}__lag{edge.lag}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        paths.append(out_path)
    return paths
