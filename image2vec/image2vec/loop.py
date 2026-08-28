"""Outer generate → render → critique loop."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai import BinaryContent

from image2vec import vision
from image2vec.prompts import critic_user_text, generator_user_text
from image2vec.render import write_png
from image2vec.schemas import ConversionResult, Critique, IterationRecord, VectorDraft
from image2vec.workspace import Workspace

ProgressFn = Callable[[str], None]


def convert_image(
    source: str | Path,
    *,
    output: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    style: str = "flat",
    max_iterations: int = 5,
    min_score: float = 0.75,
    model: str | None = None,
    generator: Any | None = None,
    critic: Any | None = None,
    progress: ProgressFn | None = print,
) -> ConversionResult:
    """Convert a raster image to SVG, stopping when the critic passes.

    `generator` and `critic` are injectable pydantic-ai agents (or fakes with
    `run_sync`). When omitted, OpenRouter-backed agents are built.
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    root = (
        Path(workspace_dir).expanduser().resolve()
        if workspace_dir
        else Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    workspace = _stage(source_path, root)
    _log(progress, f"workspace {workspace.root}  {workspace.width}x{workspace.height}  style={style}")

    if generator is None or critic is None:
        from image2vec.agents import build_critic, build_generator

        generator = generator or build_generator(workspace, model)
        critic = critic or build_critic(model)

    source_preview = _image_part(workspace.source_png)
    edges_preview = _image_part(workspace.edges_png)

    history: list[IterationRecord] = []
    last_critique: Critique | None = None
    last_render: Path | None = None

    for index in range(1, max_iterations + 1):
        last_metrics = history[-1].metrics if history else None
        user = generator_user_text(
            width=workspace.width,
            height=workspace.height,
            style=style,
            iteration=index,
            max_iterations=max_iterations,
            metrics=last_metrics,
            critique=last_critique,
        )
        prompt: list[str | BinaryContent] = [
            user,
            "SOURCE:",
            source_preview,
            "CANNY EDGES:",
            edges_preview,
        ]
        if last_render is not None:
            prompt.extend(["PREVIOUS RENDER:", _image_part(last_render)])

        draft: VectorDraft = generator.run_sync(prompt, deps=workspace).output
        record = _snapshot_iteration(workspace, index, draft, critic, style, source_preview)
        history.append(record)
        last_critique = record.critique
        last_render = workspace.resolve(record.png_path)
        _log(
            progress,
            f"iter {index}/{max_iterations}  {record.metrics.summary()}  "
            f"score={record.critique.score:.2f}  passed={record.critique.passed}",
        )
        if record.critique.passed and record.critique.score >= min_score:
            break

    return _finalize(workspace, history, output, min_score)


def _stage(source: Path, root: Path) -> Workspace:
    root.mkdir(parents=True, exist_ok=True)
    width, height = vision.flatten_to_png(source, root / "source.png")
    workspace = Workspace(root=root, width=width, height=height)
    bgr = vision.load_bgr(workspace.source_png)
    edges, _ = vision.detect_edges(bgr)
    workspace.write_bytes("edges.png", vision.encode_png(edges))
    workspace.write_text("palette.json", json.dumps(vision.extract_palette(bgr), indent=2))
    return workspace


def _snapshot_iteration(
    workspace: Workspace,
    index: int,
    draft: VectorDraft,
    critic: Any,
    style: str,
    source_preview: BinaryContent,
) -> IterationRecord:
    svg_text = workspace.resolve(draft.svg_path).read_text(encoding="utf-8")
    stem = workspace.iteration_stem(index)
    svg_rel = f"{stem}.svg"
    png_rel = f"{stem}.png"
    workspace.write_text(svg_rel, svg_text)
    png_path = write_png(
        svg_text,
        workspace.resolve(png_rel),
        width=workspace.width,
        height=workspace.height,
    )
    metrics = vision.measure_similarity(
        vision.load_bgr(workspace.source_png),
        vision.load_bgr(png_path),
    )
    workspace.write_text(f"{stem}.metrics.json", metrics.model_dump_json(indent=2))
    critique: Critique = critic.run_sync(
        [
            critic_user_text(style=style, metrics=metrics, generator_notes=draft.notes),
            "SOURCE:",
            source_preview,
            "RENDERED SVG:",
            _image_part(png_path),
        ]
    ).output
    workspace.write_text(f"{stem}.critique.json", critique.model_dump_json(indent=2))
    return IterationRecord(
        index=index,
        svg_path=svg_rel,
        png_path=png_rel,
        metrics=metrics,
        critique=critique,
        draft=draft,
    )


def _finalize(
    workspace: Workspace,
    history: list[IterationRecord],
    output: str | Path | None,
    min_score: float,
) -> ConversionResult:
    best = history[-1]
    svg_text = workspace.resolve(best.svg_path).read_text(encoding="utf-8")
    workspace.final_svg.write_text(svg_text, encoding="utf-8")
    write_png(
        svg_text,
        workspace.final_png,
        width=workspace.width,
        height=workspace.height,
    )
    if output is not None:
        dest = Path(output).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace.final_svg, dest)
    passed = best.critique.passed and best.critique.score >= min_score
    return ConversionResult(
        passed=passed,
        iterations=len(history),
        output_svg=str(workspace.final_svg),
        output_png=str(workspace.final_png),
        workspace=str(workspace.root),
        last_metrics=best.metrics,
        last_critique=best.critique,
        history=history,
    )


def _image_part(path: Path) -> BinaryContent:
    return BinaryContent(data=vision.preview_png(path), media_type="image/png")


def _log(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)
