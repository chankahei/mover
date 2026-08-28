"""Function tools the generator (and critic) can call inside a workspace."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from pydantic_ai import BinaryContent, ModelRetry, RunContext, ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from image2vec import vision
from image2vec.render import write_png
from image2vec.svg import parse_svg, prepare_svg, sanitize_tree, svg_stats
from image2vec.workspace import Workspace

VISION_TOOLS = FunctionToolset(max_retries=3)


def _bgr(ctx: RunContext[Workspace], path: str):
    try:
        resolved = ctx.deps.resolve(path)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    if not resolved.is_file():
        raise ModelRetry(f"file not found: {path}")
    return resolved, vision.load_bgr(resolved)


def _preview(path) -> BinaryContent:
    return BinaryContent(data=vision.preview_png(path), media_type="image/png")


@VISION_TOOLS.tool
def read_image(ctx: RunContext[Workspace], path: str) -> ToolReturn:
    """Read an image in the workspace and return size plus a preview.

    Args:
        path: Workspace-relative image path, e.g. source.png or iter_01.png.
    """
    resolved, _ = _bgr(ctx, path)
    return ToolReturn(
        return_value=vision.inspect_image(resolved),
        content=[f"Preview of {path}:", _preview(resolved)],
    )


@VISION_TOOLS.tool
def detect_edges(
    ctx: RunContext[Workspace],
    path: str = "source.png",
    output_path: str = "edges.png",
    low: int = 60,
    high: int = 160,
) -> ToolReturn:
    """Run OpenCV Canny edge detection and save the edge map.

    Args:
        path: Image to trace.
        output_path: Where to write the edge PNG.
        low: Canny low threshold.
        high: Canny high threshold.
    """
    _, bgr = _bgr(ctx, path)
    edges, stats = vision.detect_edges(bgr, low=low, high=high)
    dest = ctx.deps.write_bytes(output_path, vision.encode_png(edges))
    stats = {**stats, "path": output_path, "source": path}
    return ToolReturn(
        return_value=stats,
        content=[f"Canny edges for {path} → {output_path}:", _preview(dest)],
    )


@VISION_TOOLS.tool
def extract_palette(
    ctx: RunContext[Workspace],
    path: str = "source.png",
    k: int = 6,
) -> str:
    """Cluster dominant colors with OpenCV k-means.

    Args:
        path: Image to sample.
        k: Number of palette colors (2-12).
    """
    _, bgr = _bgr(ctx, path)
    palette = vision.extract_palette(bgr, k=k)
    return json.dumps(palette)


@VISION_TOOLS.tool
def trace_contours(
    ctx: RunContext[Workspace],
    path: str = "source.png",
    max_contours: int = 40,
    epsilon_ratio: float = 0.012,
) -> str:
    """Approximate OpenCV contours as SVG path `d` strings (scaffold only).

    Args:
        path: Image to trace.
        max_contours: Keep the largest N contours.
        epsilon_ratio: approxPolyDP epsilon as a fraction of arc length.
    """
    _, bgr = _bgr(ctx, path)
    paths = vision.trace_contours(bgr, max_contours=max_contours, epsilon_ratio=epsilon_ratio)
    return json.dumps({"count": len(paths), "path_d": paths})


@VISION_TOOLS.tool
def write_svg(ctx: RunContext[Workspace], svg: str, path: str = "current.svg") -> dict:
    """Validate, sanitize, and write an SVG document into the workspace.

    Args:
        svg: Complete <svg>...</svg> markup (markdown fences are stripped).
        path: Workspace-relative destination, usually current.svg.
    """
    try:
        document = prepare_svg(svg, ctx.deps.width, ctx.deps.height)
        root = sanitize_tree(parse_svg(document))
    except (ValueError, OSError, ET.ParseError) as exc:
        raise ModelRetry(f"SVG is not usable: {exc}. Return a complete <svg> document.") from exc
    ctx.deps.write_text(path, document)
    return {"path": path, "bytes": len(document.encode("utf-8")), **svg_stats(root)}


@VISION_TOOLS.tool
def render_svg(
    ctx: RunContext[Workspace],
    svg_path: str = "current.svg",
    png_path: str = "current.png",
) -> ToolReturn:
    """Rasterize an SVG in the workspace to PNG at the source pixel size.

    Args:
        svg_path: SVG to render.
        png_path: Destination PNG.
    """
    try:
        svg = ctx.deps.read_bytes(svg_path).decode("utf-8")
    except FileNotFoundError:
        raise ModelRetry(f"SVG not found: {svg_path}. Call write_svg first.") from None
    dest = ctx.deps.resolve(png_path)
    try:
        write_png(svg, dest, width=ctx.deps.width, height=ctx.deps.height)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelRetry(f"SVG failed to render: {exc}") from exc
    return ToolReturn(
        return_value={"png_path": png_path, "svg_path": svg_path},
        content=[f"Render of {svg_path}:", _preview(dest)],
    )


@VISION_TOOLS.tool
def measure_similarity(
    ctx: RunContext[Workspace],
    candidate_path: str = "current.png",
    source_path: str = "source.png",
    diff_path: str = "diff.png",
) -> ToolReturn:
    """Compare two images: SSIM, PSNR, histogram correlation, Canny edge IoU.

    Args:
        candidate_path: Rendered SVG PNG (or any candidate).
        source_path: Ground-truth raster.
        diff_path: Where to write an absolute-difference heatmap.
    """
    source_file, source = _bgr(ctx, source_path)
    try:
        candidate_file, candidate = _bgr(ctx, candidate_path)
    except ModelRetry:
        raise ModelRetry(
            f"candidate image missing: {candidate_path}. Call render_svg first."
        ) from None
    report = vision.measure_similarity(source, candidate)
    heat = vision.difference_map(source, candidate)
    dest = ctx.deps.write_bytes(diff_path, vision.encode_png(heat))
    payload = report.model_dump()
    payload["summary"] = report.summary()
    payload["diff_path"] = diff_path
    return ToolReturn(
        return_value=payload,
        content=[
            f"Source {source_file.name} vs {candidate_file.name}. Diff heatmap:",
            _preview(dest),
        ],
    )
