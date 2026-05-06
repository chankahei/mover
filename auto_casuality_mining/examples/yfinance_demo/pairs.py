"""For every edge in the curated graph, render a timeseries pair plot.

The plot overlays:

  - The (lagged) source series on the left axis,
  - The target series on the right axis,

so a human can eyeball whether the discovered causal direction looks plausible.
For VECTOR sources, the cross-column mean is plotted (the curation pipeline
already collapsed each VECTOR series into its constituent lagged columns
internally; here we just need something representable as one line).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from causality_mining import CausalGraph, TimeSeriesCollection, TimeSeriesKind


def _series_as_line(coll: TimeSeriesCollection, series_id: str) -> pd.Series:
    """Reduce any series kind to a single 1-D pd.Series for plotting."""
    ts = coll.get(series_id)
    if ts.kind is TimeSeriesKind.VECTOR:
        return ts.data.mean(axis=1).rename(series_id)
    if ts.kind is TimeSeriesKind.CATEGORICAL:
        codes, _ = pd.factorize(ts.data)
        return pd.Series(codes, index=ts.data.index, name=series_id, dtype=float)
    return ts.data.rename(series_id)


def render_edge_pairs(
    graph: CausalGraph,
    collection: TimeSeriesCollection,
    out_dir: Path,
) -> list[Path]:
    """Write one PNG per promoted edge. Returns the list of paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for edge in tqdm(
        graph.edges.values(),
        total=len(graph.edges),
        desc="viz: edge pair plots",
        unit="edge",
    ):
        src = _series_as_line(collection, edge.source).shift(edge.lag)
        tgt = _series_as_line(collection, edge.target)

        fig, ax_left = plt.subplots(figsize=(10, 4))
        ax_right = ax_left.twinx()
        ax_left.plot(src.index, src.values, color="#1f77b4", label=f"{edge.source} (lag {edge.lag})")
        ax_right.plot(tgt.index, tgt.values, color="#d35400", label=edge.target)

        ax_left.set_ylabel(f"{edge.source} (lag {edge.lag})", color="#1f77b4")
        ax_right.set_ylabel(edge.target, color="#d35400")
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
