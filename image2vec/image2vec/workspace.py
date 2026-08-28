"""Sandboxed conversion workspace: paths stay inside a single root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Workspace:
    """Files for one conversion run. All agent I/O is rooted here."""

    root: Path
    width: int
    height: int

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def source_png(self) -> Path:
        return self.root / "source.png"

    @property
    def edges_png(self) -> Path:
        return self.root / "edges.png"

    @property
    def current_svg(self) -> Path:
        return self.root / "current.svg"

    @property
    def final_svg(self) -> Path:
        return self.root / "final.svg"

    @property
    def final_png(self) -> Path:
        return self.root / "final.png"

    def resolve(self, relative: str) -> Path:
        """Resolve a workspace-relative path. Rejects traversal outside root."""
        rel = relative.strip()
        if not rel or rel.startswith("/"):
            raise ValueError(f"path must be workspace-relative: {relative!r}")
        root = self.root
        path = (root / rel).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"path escapes workspace: {relative}")
        return path

    def write_text(self, relative: str, text: str) -> Path:
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_bytes(self, relative: str, data: bytes) -> Path:
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read_bytes(self, relative: str) -> bytes:
        path = self.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        return path.read_bytes()

    def iteration_stem(self, index: int) -> str:
        return f"iter_{index:02d}"
