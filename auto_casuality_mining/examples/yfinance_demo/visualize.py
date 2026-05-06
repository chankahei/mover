"""Render the discovered `CausalGraph` with Mermaid (and save `.mmd` source)."""
from __future__ import annotations

from pathlib import Path

from mermaid import Graph, Mermaid

from causality_mining import CausalGraph


def render_graph(graph: CausalGraph, out_path: Path) -> Path:
    """Render graph to PNG via mermaid-py. Always saves `.mmd` source."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_source = _to_mermaid_script(graph)
    mmd_path = out_path.with_suffix(".mmd")
    mmd_path.write_text(mermaid_source)

    try:
        Mermaid(
            Graph(title="Discovered causal graph", script=mermaid_source),
            width=2200,
            height=1400,
        ).to_png(out_path)
        return out_path
    except Exception as exc:
        # Mermaid rendering may fail in restricted/proxied environments.
        error_path = out_path.with_suffix(".render_error.txt")
        error_path.write_text(
            "Mermaid PNG rendering failed.\n"
            f"error={exc}\n"
            f"mermaid_source={mmd_path}\n"
        )
        return mmd_path


def _to_mermaid_script(graph: CausalGraph) -> str:
    node_alias = {node_id: f"n{i}" for i, node_id in enumerate(sorted(graph.nodes))}
    lines = [
        "flowchart LR",
        "classDef target fill:#ffe8cc,stroke:#d35400,stroke-width:2px;",
        "classDef feature fill:#e8f1ff,stroke:#1f77b4,stroke-width:1px;",
    ]
    for node_id, node in graph.nodes.items():
        alias = node_alias[node_id]
        lines.append(f'{alias}["{node_id}"]')
        css = "target" if node.is_target else "feature"
        lines.append(f"class {alias} {css};")
    for edge in graph.edges.values():
        src = node_alias[edge.source]
        dst = node_alias[edge.target]
        label = (
            f"lag={edge.lag}<br/>"
            f"conf={edge.confidence:.2f}<br/>"
            f"strength={edge.strength:+.4f}"
        )
        lines.append(f'{src} -->|"{label}"| {dst}')
    return "\n".join(lines)
