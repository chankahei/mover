"""OpenCV helpers: load, edges, palette, contours, similarity, previews."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image2vec.schemas import SimilarityReport

PREVIEW_MAX_SIDE = 768
WORKING_MAX_SIDE = 1024


def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_png(bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to encode PNG")
    return encoded.tobytes()


def preview_png(path: Path, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    bgr = load_bgr(path)
    return encode_png(_fit(bgr, max_side))


def inspect_image(path: Path) -> dict[str, int | str]:
    bgr = load_bgr(path)
    h, w = bgr.shape[:2]
    return {"path": str(path.name), "width": w, "height": h, "channels": int(bgr.shape[2])}


def detect_edges(
    bgr: np.ndarray,
    low: int = 60,
    high: int = 160,
) -> tuple[np.ndarray, dict[str, float | int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [float(cv2.contourArea(c)) for c in contours] if contours else [0.0]
    stats = {
        "edge_pixel_ratio": float(np.mean(edges > 0)),
        "contour_count": len(contours),
        "largest_contour_area": float(max(areas)),
    }
    return edges, stats


def extract_palette(bgr: np.ndarray, k: int = 6) -> list[dict[str, str | float]]:
    k = max(2, min(k, 12))
    pixels = bgr.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 8000:
        idx = np.linspace(0, len(pixels) - 1, 8000).astype(int)
        pixels = pixels[idx]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float64)
    shares = counts / max(counts.sum(), 1.0)
    order = np.argsort(-shares)
    palette: list[dict[str, str | float]] = []
    for i in order:
        b, g, r = (max(0, min(255, round(float(c)))) for c in centers[i])
        palette.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "share": round(float(shares[i]), 4)})
    return palette


def trace_contours(
    bgr: np.ndarray,
    *,
    max_contours: int = 40,
    epsilon_ratio: float = 0.012,
) -> list[str]:
    edges, _ = detect_edges(bgr)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
    paths: list[str] = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
        if len(approx) < 3:
            continue
        points = " ".join(f"{int(pt[0][0])},{int(pt[0][1])}" for pt in approx)
        paths.append(f"M {points} Z")
    return paths


def difference_map(source: np.ndarray, render: np.ndarray) -> np.ndarray:
    a, b = _align(source, render)
    diff = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)


def measure_similarity(source: np.ndarray, render: np.ndarray) -> SimilarityReport:
    a, b = _align(source, render)
    h, w = a.shape[:2]
    err = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    psnr = 10.0 * math.log10((255.0 ** 2) / (float(err) + 1e-8))
    ssim = _ssim_gray(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    hist = _histogram_correlation(a, b)
    edge_iou = _edge_iou(a, b)
    return SimilarityReport(
        width=w,
        height=h,
        mse=float(err),
        psnr=float(psnr),
        ssim=float(max(0.0, min(1.0, ssim))),
        histogram_correlation=float(hist),
        edge_iou=float(edge_iou),
    )


def flatten_to_png(source: Path, dest: Path) -> tuple[int, int]:
    """Copy the input onto a white background as workspace/source.png."""
    rgba = Image.open(source).convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(canvas, rgba).convert("RGB")
    if max(rgb.size) > WORKING_MAX_SIDE:
        rgb.thumbnail((WORKING_MAX_SIDE, WORKING_MAX_SIDE), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(dest, format="PNG")
    return rgb.size


def _fit(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return bgr
    scale = max_side / long_side
    return cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return a, b


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a64, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b64, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a64 ** 2, (11, 11), 1.5) - mu_a ** 2
    sigma_b = cv2.GaussianBlur(b64 ** 2, (11, 11), 1.5) - mu_b ** 2
    sigma_ab = cv2.GaussianBlur(a64 * b64, (11, 11), 1.5) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    )
    return float(np.mean(ssim_map))


def _histogram_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([a_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([b_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    ea, _ = detect_edges(a)
    eb, _ = detect_edges(b)
    inter = np.logical_and(ea > 0, eb > 0).sum()
    union = np.logical_or(ea > 0, eb > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)
