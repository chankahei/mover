"""OpenRouter model factory for pydantic-ai agents."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import find_dotenv, load_dotenv

DEFAULT_MODEL = "openai/gpt-4o"


def load_environment() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def model_name(override: str | None = None) -> str:
    return (override or os.getenv("OPENROUTER_CHAT_MODEL") or DEFAULT_MODEL).strip()


def require_api_key() -> str:
    load_environment()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required. Copy .env.example to .env and set the key."
        )
    return key


@lru_cache(maxsize=4)
def build_model(name: str, temperature: float, max_tokens: int):
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    api_key = require_api_key()
    provider_kwargs: dict[str, str] = {"api_key": api_key}
    if os.getenv("OPENROUTER_APP_URL"):
        provider_kwargs["app_url"] = os.environ["OPENROUTER_APP_URL"]
    provider_kwargs["app_title"] = os.getenv("OPENROUTER_APP_TITLE", "image2vec")
    provider = OpenRouterProvider(**provider_kwargs)
    settings = OpenRouterModelSettings(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return OpenRouterModel(name, provider=provider), settings
