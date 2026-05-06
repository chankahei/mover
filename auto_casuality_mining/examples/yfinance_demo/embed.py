"""Tiny embeddings client.

Hits an OpenAI-compatible `/embeddings` endpoint -- by default OpenRouter, but
the base URL and model are env-configurable so a user can point this at OpenAI
directly if they prefer.

Required env vars (one of):
    OPENROUTER_API_KEY  (used if OPENROUTER_BASE_URL is the default)
    EMBED_API_KEY       (overrides OPENROUTER_API_KEY)

Optional:
    OPENROUTER_EMBED_MODEL  (default: openai/text-embedding-3-small)
    EMBED_BASE_URL          (default: https://openrouter.ai/api/v1)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


@dataclass
class EmbeddingClient:
    """Cached embedding client. Returns full-dim vectors via `embed`."""

    cache_dir: Path
    model: str = os.environ.get("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
    base_url: str = os.environ.get("EMBED_BASE_URL", "https://openrouter.ai/api/v1")

    def __post_init__(self) -> None:
        self.api_key = os.environ.get("EMBED_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("EMBED_API_KEY (or OPENROUTER_API_KEY) is required")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.Lock()

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha256(f"{self.model}:{text}".encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def embed(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embed(text) expects a non-empty string")
        cache_path = self._cache_path(text)
        with self._cache_lock:
            if cache_path.exists():
                return np.array(json.loads(cache_path.read_text())["v"], dtype=float)

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(
                f"embedding request failed ({resp.status_code}): {resp.text[:400]}"
            )
        vec = resp.json()["data"][0]["embedding"]
        with self._cache_lock:
            if not cache_path.exists():
                cache_path.write_text(json.dumps({"v": vec}))
        return np.asarray(vec, dtype=float)
