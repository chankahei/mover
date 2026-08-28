from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from image2vec.loop import convert_image
from image2vec.render import svg_to_png
from image2vec.schemas import Critique, VectorDraft

SIMPLE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
  <rect width="24" height="24" fill="#1a73e8"/>
</svg>"""


def _cairo_works() -> bool:
    try:
        svg_to_png(SIMPLE, width=24, height=24)
        return True
    except (OSError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _cairo_works(), reason="cairosvg/cairo not available")


class ScriptedAgent:
    def __init__(self, outputs: list[object]) -> None:
        self._outputs = list(outputs)
        self.prompts: list[object] = []

    def run_sync(self, prompt, deps=None):
        self.prompts.append(prompt)
        if deps is not None and not (deps.root / "current.svg").exists():
            deps.write_text("current.svg", SIMPLE)
        output = self._outputs.pop(0)
        if callable(output):
            output = output(deps)
        return SimpleNamespace(output=output)


def _source_png(path: Path) -> Path:
    Image.new("RGB", (24, 24), (26, 115, 232)).save(path)
    return path


def test_loop_stops_when_critic_passes(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "in.png")
    generator = ScriptedAgent(
        [VectorDraft(svg_path="current.svg", focus="fill", notes="solid blue")]
    )
    critic = ScriptedAgent(
        [Critique(passed=True, score=0.91, summary="Matches the blue field.", issues=[])]
        + [Critique(passed=False, score=0.1, summary="should not run", issues=[])]
    )
    result = convert_image(
        source,
        output=tmp_path / "out.svg",
        workspace_dir=tmp_path / "ws",
        style="flat",
        max_iterations=4,
        min_score=0.75,
        generator=generator,
        critic=critic,
        progress=None,
    )
    assert result.passed
    assert result.iterations == 1
    assert (tmp_path / "out.svg").is_file()
    assert "1a73e8" in (tmp_path / "out.svg").read_text()
    assert critic._outputs  # second critique unused


def test_loop_retries_until_pass(tmp_path: Path) -> None:
    source = _source_png(tmp_path / "in.png")
    generator = ScriptedAgent(
        [
            VectorDraft(svg_path="current.svg", focus="block", notes="v1"),
            VectorDraft(svg_path="current.svg", focus="color", notes="v2"),
        ]
    )
    critic = ScriptedAgent(
        [
            Critique(passed=False, score=0.4, summary="wrong", issues=[]),
            Critique(passed=True, score=0.8, summary="ok", issues=[]),
        ]
    )
    result = convert_image(
        source,
        workspace_dir=tmp_path / "ws",
        max_iterations=5,
        generator=generator,
        critic=critic,
        progress=None,
    )
    assert result.passed
    assert result.iterations == 2
    assert (tmp_path / "ws" / "iter_01.svg").is_file()
    assert (tmp_path / "ws" / "iter_02.svg").is_file()
