"""End-to-end demo: fetch -> curate -> predict -> visualize."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from tqdm.auto import tqdm

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from env import load as _load_env  # noqa: E402

_load_env()

from causality_mining import DiscoveryConfig, discover_graph  # noqa: E402
from causality_mining.normalize.pct_change import PctChange  # noqa: E402
from causality_mining.normalize.pipeline import Pipeline  # noqa: E402

_NORMALIZE_SUFFIX = "__pct_change"

from data import MAG7, build_market_collection  # noqa: E402
from ctr import build_ctr  # noqa: E402
from explode import explode_vector_series  # noqa: E402
from news import build_news_timeseries  # noqa: E402
from visualize import render_graph  # noqa: E402
from pairs import render_edge_pairs  # noqa: E402
from stimulate import (  # noqa: E402
    build_price_volume_stimulus,
    predict_multi,
    save_stimulation_report,
)


def _maybe_add_news(collection, market_index, cache_dir: Path) -> None:
    """Add the LLM-driven news VECTOR series to the collection if env permits."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("      OPENROUTER_API_KEY not set -- skipping news embedding step.")
        return
    print("      building news timeseries via OpenRouter LLM + embeddings ...")
    news = build_news_timeseries(MAG7, market_index, cache_dir=cache_dir)
    collection.add(news)
    print(f"      news vector dim = {news.data.shape[1]}")


def _should_explode_vector_nodes() -> bool:
    value = os.environ.get("YF_DEMO_EXPLODE_VECTOR_NODES", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def main() -> int:
    out_dir = _DEMO_DIR / "out"
    out_dir.mkdir(exist_ok=True)
    cache_dir = _DEMO_DIR / "cache"

    with tqdm(total=4, desc="demo pipeline", unit="step") as pipeline:
        print("[1/4] downloading market data ...")
        collection, closes, volumes = build_market_collection(start="2023-01-01", cache_dir=cache_dir)
        for ctr_ts in build_ctr(closes, volumes):
            collection.add(ctr_ts)
        _maybe_add_news(collection, closes.index, cache_dir)
        if _should_explode_vector_nodes():
            collection = explode_vector_series(collection, series_ids=("prices", "volumes"))
            print("      exploded vector nodes: prices/volumes -> per-ticker scalar nodes")
            print(f"      total timeseries after explode: {len(collection)}")

        print(f"      collected {len(collection)} timeseries:")
        for ts in collection:
            shape = ts.data.shape if hasattr(ts.data, "shape") else (len(ts.data),)
            marker = " [pre_normalized]" if ts.pre_normalized else ""
            print(f"        - {ts.id:<8} kind={ts.kind.value:<11} shape={shape}{marker}")
        pipeline.update(1)

        print("[2/4] discovering causal graph (leave-one-out LightGBM + Tree SHAP) ...")
        normalize = Pipeline(steps=(PctChange(),))
        edge_min_imp = _get_env_float("YF_DEMO_EDGE_MIN_IMPORTANCE", 0.0)
        edge_min_conf = _get_env_float("YF_DEMO_EDGE_MIN_CONFIDENCE", 0.25)
        edge_topk = _get_env_int("YF_DEMO_EDGE_TOPK_PER_TARGET", 2)
        dowhy_topk = _get_env_int("YF_DEMO_DOWHY_TOPK_PER_TARGET", 0)
        dowhy_min_conf = _get_env_float("YF_DEMO_DOWHY_MIN_CONFIDENCE", 0.25)
        cfg = DiscoveryConfig(
            freq="B",
            normalize=normalize,
            lags=(1, 2, 4, 8, 16, 32),
            importance_threshold=edge_min_imp,
            min_confidence=edge_min_conf,
            max_parents_per_target=edge_topk,
            dowhy_refute_top_k_per_target=dowhy_topk,
            dowhy_confidence_threshold=dowhy_min_conf,
            debug=True,
        )
        print(
            f"      edge filters: min_imp={edge_min_imp:.4f}, "
            f"min_conf={edge_min_conf:.2f}, topk_per_target={edge_topk}"
        )
        print(
            f"      dowhy filters: topk_per_target={dowhy_topk}, "
            f"min_conf={dowhy_min_conf:.2f}"
        )
        graph = discover_graph(collection, config=cfg)
        normalized_collection = normalize.apply_collection(collection)
        print(f"      discovered graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        for e in graph.edges.values():
            print(
                f"        edge {e.source} -> {e.target}  "
                f"lag={e.lag}  strength={e.strength:+.4f}  imp={e.importance:.4f}  conf={e.confidence:.3f}"
            )
        pipeline.update(1)

        print("[3/4] running inference: fake price + volume pct_change -> per-ticker ctr_pct_change ...")
        last_ts = closes.index[-1]
        price_delta = float(os.environ.get("YF_DEMO_PRICE_DELTA", "0.05"))
        volume_delta = float(os.environ.get("YF_DEMO_VOLUME_DELTA", "0.50"))
        stimuli = build_price_volume_stimulus(
            tickers=tuple(closes.columns),
            price_delta=price_delta,
            volume_delta=volume_delta,
            suffix=_NORMALIZE_SUFFIX,
        )
        print(
            f"      stimulus (pct-change space): prices += {price_delta:+.4f} "
            f"({price_delta * 100:+.1f}%), volumes += {volume_delta:+.4f} "
            f"({volume_delta * 100:+.1f}%) per ticker"
        )
        summaries = predict_multi(
            graph=graph,
            stimuli=stimuli,
            history=normalized_collection,
            timestamp=last_ts,
        )
        stim_report_path = save_stimulation_report(out_dir / "stimulate.txt", stimuli, summaries)
        ctr_keys = sorted(k for k in summaries if k.startswith("ctr_"))
        if not ctr_keys:
            print("      no ctr_<ticker> targets received incoming effects")
        else:
            print("      per-ticker CTR predictions:")
            for key in ctr_keys:
                ctr = summaries[key]
                print(
                    f"        {key:<24} baseline={ctr.baseline:+.4f}  "
                    f"delta={ctr.delta:+.4f}  predicted={ctr.predicted:+.4f}  conf={ctr.confidence:.2f}"
                )
        for tid, summary in sorted(summaries.items(), key=lambda kv: -abs(kv[1].delta)):
            if tid in ctr_keys:
                continue
            print(
                f"      target {tid:<24} baseline={summary.baseline:+.4f}  "
                f"delta={summary.delta:+.4f}  predicted={summary.predicted:+.4f}  "
                f"conf={summary.confidence:.2f}"
            )
        print(f"      wrote {stim_report_path}")
        pipeline.update(1)

        print("[4/4] writing visualizations ...")
        graph_artifact = render_graph(graph, out_dir / "graph.png")
        edge_paths = render_edge_pairs(graph, normalized_collection, out_dir / "edges")
        print(f"      wrote {graph_artifact}")
        print(f"      wrote {(out_dir / 'graph.mmd')}")
        print(f"      wrote {len(edge_paths)} edge pair plots into {out_dir / 'edges'}")
        pipeline.update(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
