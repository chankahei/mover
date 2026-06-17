"""LLM access via pydantic-ai.

A single configured model is shared by every agent. Configuration is read from
the environment so the same code works against OpenAI or any compatible gateway:

- ``OPENAI_API_KEY``  - API key
- ``BASE_URL``        - optional OpenAI-compatible base URL
- ``GUARDRAIL_LLM_MODEL`` (falls back to ``DOGE_LLM_MODEL`` then a default)

``build_agent`` returns a typed :class:`pydantic_ai.Agent`: pass the Pydantic
``output_type`` you want back and pydantic-ai handles the structured decoding
and validation. ``run`` is a thin synchronous helper around ``agent.run_sync``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Type, TypeVar

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

_DEFAULT_MODEL = "gpt-4o-mini"

T = TypeVar("T")


def _model_name() -> str:
    return os.getenv("GUARDRAIL_LLM_MODEL") or os.getenv("DOGE_LLM_MODEL") or _DEFAULT_MODEL


@lru_cache(maxsize=1)
def _model() -> OpenAIChatModel:
    kwargs: dict[str, str] = {}
    if os.getenv("BASE_URL"):
        kwargs["base_url"] = os.environ["BASE_URL"]
    if os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    return OpenAIChatModel(_model_name(), provider=OpenAIProvider(**kwargs))


def build_agent(output_type: Type[T], system_prompt: str) -> Agent[None, T]:
    """A typed agent that returns ``output_type`` validated by Pydantic."""
    return Agent(_model(), output_type=output_type, system_prompt=system_prompt)


def run(agent: Agent[None, T], user_prompt: str) -> T:
    return agent.run_sync(user_prompt).output
