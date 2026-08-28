from pathlib import Path

import numpy as np
from PIL import Image

from image2vec import vision


def _solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_identical_images_are_similar(tmp_path: Path) -> None:
    path = _solid(tmp_path / "a.png", (40, 80, 160))
    bgr = vision.load_bgr(path)
    report = vision.measure_similarity(bgr, bgr.copy())
    assert report.ssim > 0.99
    assert report.histogram_correlation > 0.99
    assert report.mse < 1e-6


def test_different_colors_drop_histogram(tmp_path: Path) -> None:
    red = vision.load_bgr(_solid(tmp_path / "r.png", (220, 20, 20)))
    blue = vision.load_bgr(_solid(tmp_path / "b.png", (20, 20, 220)))
    report = vision.measure_similarity(red, blue)
    assert report.histogram_correlation < 0.5
    assert report.ssim < 0.95


def test_edges_on_contrast_block() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :32] = (255, 255, 255)
    edges, stats = vision.detect_edges(image)
    assert stats["edge_pixel_ratio"] > 0
    assert edges.shape == (64, 64)


def test_palette_finds_dominant_hex() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:] = (0, 0, 255)  # BGR red
    palette = vision.extract_palette(image, k=2)
    hexes = {item["hex"] for item in palette}
    assert any(h.startswith("#") and h.lower().startswith("#ff") for h in hexes)


def test_flatten_composites_transparency(tmp_path: Path) -> None:
    src = tmp_path / "t.png"
    Image.new("RGBA", (20, 10), (255, 0, 0, 0)).save(src)
    dest = tmp_path / "out.png"
    width, height = vision.flatten_to_png(src, dest)
    assert (width, height) == (20, 10)
    rgb = np.array(Image.open(dest))
    assert tuple(rgb[0, 0]) == (255, 255, 255)
