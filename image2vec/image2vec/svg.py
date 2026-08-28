"""Parse, sanitize, and summarize SVG documents the model writes."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_SVG_NS = "http://www.w3.org/2000/svg"
_FENCE = re.compile(r"```(?:svg)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_STRIP_TAGS = {
    "script",
    "foreignObject",
    "animate",
    "animateTransform",
    "animateMotion",
    "set",
}


def extract_svg(text: str) -> str:
    """Pull a <svg>...</svg> document out of model output or markdown fences."""
    blob = text.strip()
    fenced = _FENCE.search(blob)
    if fenced:
        blob = fenced.group(1).strip()
    start = blob.lower().find("<svg")
    end = blob.lower().rfind("</svg>")
    if start < 0 or end < 0 or end < start:
        raise ValueError("no <svg>...</svg> document found")
    return blob[start : end + len("</svg>")].strip()


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_svg(svg: str) -> ET.Element:
    document = extract_svg(svg)
    root = ET.fromstring(document)
    if _local_tag(root.tag) != "svg":
        raise ValueError(f"root element is <{_local_tag(root.tag)}>, not <svg>")
    return root


def sanitize_tree(root: ET.Element) -> ET.Element:
    """Drop scripts, animation, and event-handler attributes."""
    for parent in list(root.iter()):
        for child in list(parent):
            if _local_tag(child.tag) in _STRIP_TAGS:
                parent.remove(child)
    for element in root.iter():
        for attr in list(element.attrib):
            if attr.lower().startswith("on"):
                del element.attrib[attr]
    return root


def ensure_viewbox(root: ET.Element, width: int, height: int) -> None:
    if not root.get("viewBox") and not root.get("viewbox"):
        root.set("viewBox", f"0 0 {width} {height}")
    if not root.get("width"):
        root.set("width", str(width))
    if not root.get("height"):
        root.set("height", str(height))


def _qualify_svg_ns(root: ET.Element) -> None:
    """Put the tree in the SVG namespace so serialization emits a single xmlns."""
    root.attrib.pop("xmlns", None)
    if root.tag.startswith("{"):
        return
    for element in root.iter():
        if isinstance(element.tag, str) and not element.tag.startswith("{"):
            element.tag = f"{{{_SVG_NS}}}{element.tag}"


def to_string(root: ET.Element) -> str:
    _qualify_svg_ns(root)
    ET.register_namespace("", _SVG_NS)
    payload = ET.tostring(root, encoding="unicode")
    if not payload.lstrip().startswith("<?xml"):
        payload = '<?xml version="1.0" encoding="UTF-8"?>\n' + payload
    return payload


def prepare_svg(text: str, width: int, height: int) -> str:
    root = sanitize_tree(parse_svg(text))
    ensure_viewbox(root, width, height)
    return to_string(root)


def svg_stats(root: ET.Element) -> dict[str, int | str]:
    counts: dict[str, int] = {}
    for element in root.iter():
        name = _local_tag(element.tag)
        counts[name] = counts.get(name, 0) + 1
    return {
        "elements": sum(counts.values()),
        "paths": int(counts.get("path", 0)),
        "circles": int(counts.get("circle", 0)),
        "rects": int(counts.get("rect", 0)),
        "polygons": int(counts.get("polygon", 0)),
        "viewBox": root.get("viewBox") or root.get("viewbox") or "",
    }
