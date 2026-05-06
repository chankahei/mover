# yfinance demo

Validates the `causality_mining` package on real market data.

What it does:

1. Pulls daily OHLCV for the Magnificent Seven (AAPL, MSFT, GOOGL, AMZN, META,
   NVDA, TSLA) plus `^VIX` from yfinance.
2. Pulls recent news headlines per ticker via yfinance, then for every market
   day calls an OpenRouter chat model with `(yesterday, today)` headlines to
   generate a "what changed today" report. Each report is then embedded via an
   OpenAI-compatible `/embeddings` endpoint and added as a VECTOR timeseries
   with `pre_normalized=True` (the report already encodes change semantics, so
   the package's `Delta` normalizer must NOT be applied to it).
3. Synthesizes **one fake CTR timeseries per ticker** whose true generative process is

   ```
   ctr_{ticker,t} = 0.6 * |price_return_{ticker,t}| + 0.4 * volume_growth_{ticker,t} + noise
   ```

   (note: SAME-DAY price/volume, not yesterday's). Each `ctr_<ticker>` series is
   marked `pre_normalized=True` so the discovery normalizer (see below) does
   not touch it. The package should rediscover that today's price/volume moves
   are the real causal parents of each ticker's CTR (recovered at positive lags), while VIX and
   news should be weaker.
4. Runs `discover_graph` (leave-one-out **LightGBM** + Tree SHAP). Continuous
   targets fit `LGBMRegressor`, categorical targets fit `LGBMClassifier`; SHAP
   values come from `shap.TreeExplainer`. LightGBM captures non-linear effects
   (e.g. the `|return|` term in CTR) that linear Ridge would miss completely.
5. Runs a multi-stimulus `predict_multi`: every ticker's
   `prices_<t>__pct_change` and `volumes_<t>__pct_change` get a fake pct-change
   delta injected, and the demo reports resulting `ctr_<ticker>` predictions to
   answer the question "what does the discovered graph think CTR will do given
   a market-wide price/volume shock?".
6. Renders the following artifacts under `out/`:
   - `out/graph.png` — Mermaid-rendered discovered causal graph (direction +
     `lag`, `conf`, `strength` on every edge).
   - `out/graph.mmd` — Mermaid source for the graph.
   - `out/edges/<source>__<target>__lag<l>.png` — the timeseries pair plot for
     every discovered edge.
   - `out/stimulate.txt` — saved fake-stimulus inference results.

If the environment blocks Mermaid's remote renderer, the demo still writes
`out/graph.mmd` plus `out/graph.render_error.txt`.

## Run

```bash
export OPENROUTER_API_KEY=...                                   # required for news
export OPENROUTER_CHAT_MODEL=openai/gpt-4o-mini                  # optional override
export OPENROUTER_EMBED_MODEL=openai/text-embedding-3-small      # optional override

uv run python examples/yfinance_demo/run.py
```

If `OPENROUTER_API_KEY` is not set the demo still runs, but the `news` node is
skipped. LLM responses and embeddings are cached to `examples/yfinance_demo/cache/`
so reruns are free.

Market downloads (OHLCV + VIX) and raw yfinance ticker headlines are also cached
under `examples/yfinance_demo/cache/`, so reruns for debugging avoid downloading
again by default.

To force refresh all caches:

```bash
export YF_DEMO_REFRESH_CACHE=1
```

Progress bars (`tqdm`) are shown for:

- Overall pipeline progress in `run.py`
- Ticker headline fetches
- Per-day LLM + embedding work
- Edge-pair plot rendering

Control thread-pool size with:

```bash
export YF_DEMO_MAX_WORKERS=6
```

By default, the demo explodes `prices` and `volumes` VECTOR series into
per-ticker SCALAR nodes (`prices_AAPL`, `volumes_AAPL`, etc.) before curation,
so graph topology is ticker-level rather than bundle-level.

Control this behavior with:

```bash
export YF_DEMO_EXPLODE_VECTOR_NODES=1   # default
```

Tune the size of the fake stimulus used for inference. The pipeline
normalizes prices and volumes with `PctChange`, so deltas are in
percent-change space:

```bash
export YF_DEMO_PRICE_DELTA=0.05         # default +5% price pct_change per ticker
export YF_DEMO_VOLUME_DELTA=0.50        # default +50% volume pct_change per ticker
```

Tune edge filtering (useful when `out/edges` is too dense):

```bash
export YF_DEMO_EDGE_MIN_IMPORTANCE=0.0      # default 0 (rely on relative confidence)
export YF_DEMO_EDGE_MIN_CONFIDENCE=0.25     # default
export YF_DEMO_EDGE_TOPK_PER_TARGET=2       # default
export YF_DEMO_DOWHY_TOPK_PER_TARGET=0      # default 0 (disabled; DoWhy uses linear regression which underrates LightGBM's non-linear edges)
export YF_DEMO_DOWHY_MIN_CONFIDENCE=0.25    # default
```

## Files (one job each)

```
data.py        pull yfinance OHLCV + VIX into a TimeSeriesCollection
ctr.py         synthesize the fake CTR target
explode.py     explode VECTOR series into per-column SCALAR nodes
llm.py         OpenRouter chat client (cached)
embed.py       OpenRouter-compatible embeddings client
news.py        yesterday+today headlines -> LLM change report -> embedded VECTOR series
stimulate.py   build fake price/volume stimuli + run multi-stimulus inference
visualize.py   render the causal graph with mermaid-py
pairs.py       render per-edge timeseries pair plots
run.py         entrypoint
```

## Causal discovery model

The demo uses `causality_mining.discover_graph(...)`:

- Normalize all SCALAR / VECTOR series (except those marked
  `pre_normalized=True`) with `PctChange` (i.e. `x_t / x_{t-1} - 1`). Prices,
  volumes, and VIX become unitless pct-change series; news vectors are
  pre-normalized; CTR series are pre-normalized so they stay in their native
  raw scale.
- For every SCALAR / CATEGORICAL series, predict its current value from
  lagged copies of every other series (lags `1, 2, 4, 8, 16, 32`). The panel uses business-day
  frequency (`freq="B"`) so weekend NaNs do not interact with lag dropna
  to drop most rows.
- Continuous targets use `LGBMRegressor`; categorical targets use
  `LGBMClassifier`. LightGBM captures non-linear relationships (e.g. the
  `|return|` term in the CTR generative process) that `Ridge` cannot.
- SHAP is computed via `shap.TreeExplainer`. Per-feature mean(|shap|) is
  aggregated up to `(source_series, lag)`. The best lag wins per source.
- Optional hybrid mode (off by default; enable via
  `YF_DEMO_DOWHY_TOPK_PER_TARGET>0`) runs DoWhy refutation on the top-k
  discovered parents per target. Note that DoWhy's `linear_regression`
  estimator will underweight non-linear LightGBM edges, so it is best left
  off when LightGBM is the discovery model.
- VECTOR series are NEVER targets but are still used as features. They are
  always reduced to a single column per timestamp via sign-preserving abs-max
  pooling along the column axis: if any element of the vector has a causal
  effect at a given lag, the vector is treated as having that causal effect.
