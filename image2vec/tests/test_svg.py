import pytest

from image2vec.svg import extract_svg, parse_svg, prepare_svg, sanitize_tree, svg_stats


def test_extract_from_fence() -> None:
    text = """here you go
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>
```
"""
    svg = extract_svg(text)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_parse_rejects_non_svg() -> None:
    with pytest.raises(ValueError, match="no <svg>"):
        extract_svg("<html></html>")


def test_sanitize_strips_script() -> None:
    raw = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
      <script>alert(1)</script>
      <rect width="8" height="8" onclick="x()" fill="#000"/>
    </svg>"""
    root = sanitize_tree(parse_svg(raw))
    serialized = prepare_svg(raw, 8, 8)
    assert "script" not in serialized.lower()
    assert "onclick" not in serialized.lower()
    assert root.find(".//{http://www.w3.org/2000/svg}script") is None


def test_prepare_injects_viewbox() -> None:
    raw = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="4" cy="4" r="3"/></svg>'
    out = prepare_svg(raw, 16, 9)
    assert 'viewBox="0 0 16 9"' in out
    stats = svg_stats(parse_svg(out))
    assert stats["circles"] == 1
    assert stats["elements"] >= 2
