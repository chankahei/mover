"""Multi-stimulus causal inference helper for the demo.

Build a list of fake (series_id, delta) stimuli and run `predict()` once per
stimulus, summing the resulting per-target deltas. Each stimulus's `value`
passed into the inference engine is set so the engine's
`magnitude = value - last_observed` exactly equals the desired delta.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from causality_mining import (
    CausalGraph,
    InferenceConfig,
    NewEvent,
    TimeSeriesCollection,
    predict,
)
from causality_mining.normalize.base import Change
from causality_mining.normalize.encode import encode_features


@dataclass(frozen=True)
class Stimulus:
    """A single hypothetical change to inject on one source node."""

    series_id: str
    delta: float


@dataclass(frozen=True)
class TargetSummary:
    """Aggregated multi-stimulus prediction for one target node."""

    target_id: str
    baseline: float
    delta: float
    predicted: float
    confidence: float
    n_stimuli: int


def build_price_volume_stimulus(
    tickers: Iterable[str],
    price_delta: float,
    volume_delta: float,
    suffix: str,
) -> list[Stimulus]:
    """One price + one volume stimulus per ticker, on the post-encode nodes.

    `suffix` must match the suffix the configured `Change` adds to ids when
    building the feature panel (e.g. `__pct_change_back1` for `PctChange()`).
    """
    out: list[Stimulus] = []
    for t in tickers:
        out.append(Stimulus(series_id=f"prices_{t}{suffix}", delta=price_delta))
        out.append(Stimulus(series_id=f"volumes_{t}{suffix}", delta=volume_delta))
    return out


def predict_multi(
    graph: CausalGraph,
    stimuli: list[Stimulus],
    history: TimeSeriesCollection,
    timestamp: pd.Timestamp,
    change: Change,
    freq: str = "B",
) -> dict[str, TargetSummary]:
    """Run inference for every stimulus and sum the per-target deltas.

    Each stimulus's `delta` is the magnitude of the 1-step backward change to
    inject on the source node, in the encoder's units (e.g. 0.05 = +5% pct
    change). The inference engine reads it as the treatment magnitude directly.
    """
    cfg = InferenceConfig(freq=freq, change=change)
    encoded_to_raw_pre = {
        encoded.id: raw.pre_normalized
        for raw, encoded in zip(history, encode_features(history, change))
    }

    accum: dict[str, dict] = {}
    for stim in stimuli:
        if stim.series_id not in graph.nodes:
            continue
        event = NewEvent(
            series_id=stim.series_id,
            timestamp=timestamp,
            value=stim.delta,
        )
        result = predict(graph, event=event, history=history, config=cfg)
        for tid, tp in result.targets.items():
            row = accum.setdefault(
                tid,
                {"baseline": tp.baseline, "delta": 0.0, "confidences": [], "n": 0},
            )
            row["delta"] += tp.delta
            row["confidences"].append(tp.confidence)
            row["n"] += 1

    out: dict[str, TargetSummary] = {}
    for tid, row in accum.items():
        if encoded_to_raw_pre.get(tid, False):
            predicted = row["baseline"] + row["delta"]
        else:
            predicted = change.apply(row["baseline"], row["delta"])
        out[tid] = TargetSummary(
            target_id=tid,
            baseline=row["baseline"],
            delta=row["delta"],
            predicted=predicted,
            confidence=float(np.mean(row["confidences"])) if row["confidences"] else 0.0,
            n_stimuli=row["n"],
        )
    return out


def save_stimulation_report(
    out_path: Path,
    stimuli: list[Stimulus],
    summaries: dict[str, TargetSummary],
) -> Path:
    """Write a text report of stimuli and aggregated target predictions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("Stimulation Inputs")
    lines.append("------------------")
    for s in stimuli:
        lines.append(f"{s.series_id}: delta={s.delta:+.6f}")
    lines.append("")
    lines.append("Aggregated Predictions")
    lines.append("----------------------")
    for tid, summary in sorted(summaries.items(), key=lambda kv: -abs(kv[1].delta)):
        lines.append(
            f"{tid}: baseline={summary.baseline:+.6f}  delta={summary.delta:+.6f}  "
            f"predicted={summary.predicted:+.6f}  conf={summary.confidence:.3f}  "
            f"n_stimuli={summary.n_stimuli}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
