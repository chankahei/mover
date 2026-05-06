"""Load environment variables from `.env` files.

Search order (later files override earlier ones):

  1. Repo root `.env` (the canonical project-wide secrets file).
  2. `examples/yfinance_demo/.env` (per-demo overrides, optional).

`python-dotenv` is treated as a hard dependency of the demo; install it with
`uv add python-dotenv` or `uv pip install python-dotenv` if it isn't already.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent.parent


def load() -> None:
    """Load .env files. Idempotent and safe to call multiple times."""
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_DEMO_DIR / ".env", override=True)
