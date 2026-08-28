from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image2vec.render import svg_to_png, write_png
from image2vec.svg import prepare_svg

SIMPLE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" fill="#ffffff"/>
  <circle cx="16" cy="16" r="10" fill="#cc0000"/>
</svg>"""


def _cairo_works() -> bool:
    try:
        svg_to_png(SIMPLE, width=32, height=32)
        return True
    except (OSError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _cairo_works(), reason="cairosvg/cairo not available")


def test_svg_renders_red_circle(tmp_path: Path) -> None:
    dest = write_png(SIMPLE, tmp_path / "out.png", width=32, height=32)
    rgb = np.array(Image.open(dest).convert("RGB"))
    center = rgb[16, 16]
    corner = rgb[0, 0]
    assert int(center[0]) > 150
    assert int(center[1]) < 80
    assert int(corner[0]) > 240


def test_prepare_then_render_is_png() -> None:
    document = prepare_svg(SIMPLE, 32, 32)
    png = svg_to_png(document, width=32, height=32)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
