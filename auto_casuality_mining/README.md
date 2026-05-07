# causality_mining

A small Python package for two jobs that sit at the heart of an event-driven
finance app (see `finance_app_event_driven_architecture (1).md` for product
context):

1. **Causal graph curation** — given a collection of heterogeneous timeseries
   (some of which are designated *targets* such as CTR, order ticket open,
   executed trade), build a causal graph whose edges have survived a discovery
   + refutation pipeline.
2. **Causal graph inference** — given that curated graph and a *new event* on
   one timeseries, predict the response on the target timeseries.

The package leans on [DoWhy v0.14](https://www.pywhy.org/dowhy/v0.14/) for
identification / estimation / refutation, and on pandas for number crunching.

## Input timeseries kinds

Three kinds are supported (see `causality_mining/timeseries/kind.py`):

| Kind | Storage | Example |
|---|---|---|
| `SCALAR` (1-D real) | `pd.Series[float]` | price, return, volume, dwell time |
| `CATEGORICAL` | `pd.Series[str]` | sector, regime, article frame |
| `VECTOR` (n-D real) | `pd.DataFrame[float]` | embedding, factor exposures, OHLCV |

## Change encoders

`causality_mining/normalize/` ships `Change` encoders that turn raw levels into
"change" representations. Every causal edge in this package means

> a 1-step *backward* change in the source at time t causes a `lag`-step
> *forward* change in the target between t and t+lag

so the encoder defines BOTH directions (`backward(h)` and `forward(h)`):

- `Delta` — arithmetic difference `x[b] - x[a]`
- `PctChange` — `x[b] / x[a] - 1`, scale-invariant for positive levels
- `LogReturn` — `log(x[b]) - log(x[a])`, additive in the log domain

For CATEGORICAL inputs, every encoder returns a one-hot indicator over the
`(prev_label, curr_label)` pair (see `normalize/transition.py`).

## Public API

```python
from causality_mining import (
    TimeSeries, TimeSeriesCollection, TimeSeriesKind,
    TargetSpec, curate_graph,
    NewEvent, predict,
)

graph = curate_graph(collection, targets=[TargetSpec("ctr"), TargetSpec("trade")])
result = predict(graph, event=NewEvent(series_id="news_sentiment", value=0.8, timestamp=ts),
                 history=collection)
```

The `CausalGraph` is JSON-serializable (`graph.to_json(path)` /
`CausalGraph.from_json(path)`), so the curated graph can be promoted weekly per
the architecture doc and consumed by the inference engine in production.

## File layout (one job per file)

```
causality_mining/
  timeseries/   kind.py series.py collection.py
  normalize/    base.py delta.py pct_change.py log_return.py transition.py encode.py
  panel/        resample.py builder.py
  graph/        node.py edge.py causal_graph.py io.py
  curation/     targets.py candidates.py importance.py forward_target.py loo.py refute.py curator.py discover.py
  inference/    event.py effect.py propagate.py prediction.py engine.py
```
