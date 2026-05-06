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

## Universal normalizations

`causality_mining/normalize/` ships transforms that consistently aid causal
discovery on financial / behavioral series:

- `Delta` — first difference (record the change, not the level)
- `LogReturn` — `log(x_t / x_{t-1})`, the canonical price normalization
- `ZScore` — rolling z-score (so unit changes do not dominate importance)
- `OneHot` — categorical → indicator scalars (one series per level)
- `Pipeline` — compose multiple normalizations

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
  normalize/    base.py delta.py log_return.py zscore.py one_hot.py pipeline.py
  panel/        resample.py lag.py builder.py
  graph/        node.py edge.py causal_graph.py io.py
  curation/     targets.py candidates.py importance.py refute.py curator.py
  inference/    event.py effect.py propagate.py prediction.py engine.py
```
