"""Pydantic AI generator and critic agents on OpenRouter."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai_harness import FileSystem

from image2vec.config import build_model, model_name
from image2vec.prompts import CRITIC_SYSTEM, generator_system_prompt
from image2vec.schemas import Critique, VectorDraft
from image2vec.svg import parse_svg
from image2vec.tools import VISION_TOOLS
from image2vec.workspace import Workspace

_FS_PATTERNS = (
    "*.svg",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.webp",
    "*.gif",
    "*.json",
    "*.txt",
    "*.md",
)


def build_generator(
    workspace: Workspace,
    model: str | None = None,
) -> Agent[Workspace, VectorDraft]:
    llm, settings = build_model(model_name(model), temperature=0.4, max_tokens=16384)
    agent: Agent[Workspace, VectorDraft] = Agent(
        llm,
        output_type=VectorDraft,
        deps_type=Workspace,
        system_prompt=generator_system_prompt(),
        model_settings=settings,
        toolsets=[VISION_TOOLS],
        retries=3,
        capabilities=[
            FileSystem(
                root_dir=workspace.root,
                allowed_patterns=list(_FS_PATTERNS),
                max_read_lines=15000,
            )
        ],
    )

    @agent.output_validator
    def _svg_written(ctx: RunContext[Workspace], output: VectorDraft) -> VectorDraft:
        try:
            path = ctx.deps.resolve(output.svg_path)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        if not path.is_file():
            raise ModelRetry(
                f"{output.svg_path} does not exist. Call write_svg before finishing."
            )
        try:
            parse_svg(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, ET.ParseError) as exc:
            raise ModelRetry(f"{output.svg_path} is not valid SVG: {exc}") from exc
        return output

    return agent


def build_critic(model: str | None = None) -> Agent[None, Critique]:
    llm, settings = build_model(model_name(model), temperature=0.1, max_tokens=2048)
    return Agent(
        llm,
        output_type=Critique,
        system_prompt=CRITIC_SYSTEM,
        model_settings=settings,
        retries=2,
    )
