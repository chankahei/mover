"""Rasterize SVG to PNG for visual comparison."""

from __future__ import annotations

import ctypes.util
import os
from io import BytesIO
from pathlib import Path

from PIL import Image

from image2vec.svg import prepare_svg

_CAIRO_NAMES = ("cairo-2", "cairo", "libcairo-2")


def svg_to_png(
    svg: str,
    *,
    width: int,
    height: int,
    background: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """Render SVG bytes to an RGB PNG composited on a solid background."""
    cairosvg = _load_cairosvg()
    document = prepare_svg(svg, width, height)
    png = cairosvg.svg2png(
        bytestring=document.encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    image = Image.open(BytesIO(png)).convert("RGBA")
    canvas = Image.new("RGBA", (width, height), background + (255,))
    if image.size != canvas.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return _png_bytes(Image.alpha_composite(canvas, image).convert("RGB"))


def write_png(svg: str, dest: Path, *, width: int, height: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(svg_to_png(svg, width=width, height=height))
    return dest


def _load_cairosvg():
    _patch_cairo_discovery()
    try:
        import cairosvg
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "cairosvg could not load cairo. On macOS run `brew install cairo`. "
            "You can also set CAIRO_LIB to the absolute libcairo path "
            "(e.g. /opt/homebrew/lib/libcairo.2.dylib)."
        ) from exc
    return cairosvg


def _patch_cairo_discovery() -> None:
    """Help cairocffi find Homebrew/system cairo on macOS."""
    cairo = _find_cairo_library()
    if cairo is None:
        return
    original = ctypes.util.find_library

    def find_library(name: str) -> str | None:
        if name in _CAIRO_NAMES:
            return cairo
        return original(name)

    ctypes.util.find_library = find_library  # type: ignore[method-assign]


def _find_cairo_library() -> str | None:
    explicit = os.environ.get("CAIRO_LIB", "").strip()
    candidates = [explicit] if explicit else []
    prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    candidates.extend(
        [
            f"{prefix}/lib/libcairo.2.dylib",
            f"{prefix}/opt/cairo/lib/libcairo.2.dylib",
            "/usr/local/lib/libcairo.2.dylib",
            "/opt/local/lib/libcairo.2.dylib",
        ]
    )
    for name in _CAIRO_NAMES:
        found = ctypes.util.find_library(name)
        if found:
            candidates.append(found)
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
