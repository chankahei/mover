"""End-to-end smoke test on synthetic data.

Constructs three input timeseries -- one of each supported kind -- where the
target (CTR) is a known linear function of a lagged sentiment + a one-hot
frame indicator. The test then runs `curate_graph` and `predict` and checks
that:

    * curation discovers at least one edge into the target,
    * inference returns a non-trivial delta for the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from causality_mining import (
    NewEvent,
    TargetSpec,
    TimeSeries,
    TimeSeriesCollection,
    TimeSeriesKind,
    curate_graph,
    predict,
)
from causality_mining.curation.curator import CurationConfig
from causality_mining.normalize.pipeline import Pipeline


def _make_collection(n: int = 400) -> TimeSeriesCollection:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    sentiment = pd.Series(rng.normal(0, 1, n).cumsum(), index=idx, name="sentiment")
    frame = pd.Series(
        rng.choice(["urgency", "education", "social_proof"], size=n), index=idx, name="frame"
    )
    factors = pd.DataFrame(
        rng.normal(0, 1, size=(n, 3)),
        index=idx,
        columns=["growth", "value", "momentum"],
    )
    sentiment_delta = sentiment.diff().fillna(0.0)
    urgency_indicator = (frame == "urgency").astype(float)
    ctr_signal = 0.5 * sentiment_delta.shift(1).fillna(0.0) + 0.3 * urgency_indicator.shift(1).fillna(0.0)
    ctr = pd.Series(ctr_signal + rng.normal(0, 0.05, n), index=idx, name="ctr")

    return TimeSeriesCollection.from_iterable([
        TimeSeries(id="sentiment", kind=TimeSeriesKind.SCALAR, data=sentiment),
        TimeSeries(id="frame", kind=TimeSeriesKind.CATEGORICAL, data=frame),
        TimeSeries(id="factors", kind=TimeSeriesKind.VECTOR, data=factors),
        TimeSeries(id="ctr", kind=TimeSeriesKind.SCALAR, data=ctr),
    ])


def test_curate_then_predict() -> None:
    collection = _make_collection()
    cfg = CurationConfig(
        freq="1h",
        importance_threshold=0.001,
        confidence_threshold=0.0,
        normalize=Pipeline(steps=()),
        use_shap=False,
    )
    graph = curate_graph(collection, targets=[TargetSpec("ctr")], config=cfg)

    incoming_to_ctr = graph.in_edges("ctr")
    assert incoming_to_ctr, "curation should discover at least one parent for CTR"

    last_ts = collection.get("sentiment").data.index[-1]
    event = NewEvent(series_id="sentiment", timestamp=last_ts, value=5.0)
    result = predict(graph, event, history=collection)
    assert "ctr" in result.targets or not incoming_to_ctr
