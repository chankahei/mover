from pathlib import Path

import pytest

from image2vec.workspace import Workspace


def test_resolve_stays_inside_root(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, width=10, height=10)
    path = ws.resolve("current.svg")
    assert path == tmp_path.resolve() / "current.svg"


def test_resolve_rejects_escape(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, width=10, height=10)
    with pytest.raises(ValueError, match="escapes"):
        ws.resolve("../secret.txt")


def test_resolve_rejects_absolute(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, width=10, height=10)
    with pytest.raises(ValueError, match="workspace-relative"):
        ws.resolve("/etc/passwd")


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, width=10, height=10)
    ws.write_text("notes/hello.txt", "hi")
    assert ws.read_bytes("notes/hello.txt") == b"hi"
    assert ws.iteration_stem(3) == "iter_03"
