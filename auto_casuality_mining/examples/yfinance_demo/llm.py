"""Tiny OpenRouter chat client with on-disk caching.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint
(`https://openrouter.ai/api/v1/chat/completions`). We hit it directly with
`requests` to avoid pulling in the openai SDK, and we cache every prompt to
disk so reruns of the demo never re-bill the same call.

Required env var:
    OPENROUTER_API_KEY

Optional env vars:
    OPENROUTER_CHAT_MODEL  (default: openai/gpt-4o-mini)
    OPENROUTER_BASE_URL    (default: https://openrouter.ai/api/v1)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class LLMClient:
    """Cached OpenRouter chat client."""

    cache_dir: Path
    model: str = os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
    base_url: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    def __post_init__(self) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY env var is required for the news LLM step")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.Lock()

    def _cache_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def _normalize_content(self, payload: dict[str, Any]) -> str:
        """Extract assistant text across OpenRouter/OpenAI response variants."""
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                        continue
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                text = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
                if text:
                    return text
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
            details = message.get("reasoning_details")
            if isinstance(details, list):
                detail_parts: list[str] = []
                for item in details:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        detail_parts.append(item["text"])
                text = "\n".join(p.strip() for p in detail_parts if p and p.strip()).strip()
                if text:
                    return text
        raise RuntimeError("chat response did not contain textual assistant content")

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 350,
        temperature: float = 0.2,
    ) -> str:
        """Run one chat completion. Returns the assistant message content."""
        cache_key = json.dumps(
            {"sys": system, "user": user, "model": self.model, "mt": max_tokens, "t": temperature},
            sort_keys=True,
        )
        cache_path = self._cache_path(cache_key)
        with self._cache_lock:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text()).get("content")
                if isinstance(cached, str) and cached.strip():
                    return cached

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = self._normalize_content(resp.json())
        with self._cache_lock:
            if not cache_path.exists():
                cache_path.write_text(json.dumps({"content": content}))
        return content
