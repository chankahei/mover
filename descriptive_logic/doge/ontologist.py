from __future__ import annotations

import os

from doge.env import load_environment
from doge.models import BreakdownSuggestion, LogicGate, RuleSuggestion


SYSTEM_PROMPT = """You are an ontological engineer for regulatory guardrails.
Break vague compliance requirements into binary, observable predicates.

Return only structured data matching the requested schema:
- logic_gate: AND when every condition must hold, OR when any condition is enough.
- conditions: small predicates a deterministic checker could evaluate as true/false.

Do not provide legal advice. Do not include subjective predicates unless you further
decompose them into concrete textual or data conditions.
"""


class Ontologist:
    """Pydantic AI backed suggester for Description Logic breakdowns."""

    def __init__(self, model: str | None = None) -> None:
        load_environment()
        self.model = _model_name(model or os.environ.get("DOGE_LLM_MODEL", "gpt-5.2"))
        self.base_url = _base_url()
        self._agent = None

    def suggest(self, rule_text: str) -> BreakdownSuggestion:
        agent = self._get_agent()
        prompt = (
            "Target concept:\n"
            f"{rule_text}\n\n"
            "Propose 3-6 binary child predicates using description logic style. "
            "Each predicate should be phrased as a condition that can be verified."
        )
        result = agent.run_sync(prompt)
        output = getattr(result, "output", None) or getattr(result, "data", None)
        if not isinstance(output, BreakdownSuggestion):
            raise RuntimeError("ontologist agent returned an unexpected output shape")
        return output

    def _get_agent(self):
        if self._agent is not None:
            return self._agent
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError as exc:
            raise RuntimeError(
                "pydantic-ai-slim[openai] is required for LLM suggestions. Install this "
                "package with `pip install -e .`, or run `doge build --offline` for "
                "deterministic placeholder suggestions."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for LLM suggestions. Set it in .env or run "
                "`uv run doge.py build --offline`."
            )
        provider_kwargs = {"api_key": api_key}
        if self.base_url:
            provider_kwargs["base_url"] = self.base_url
        model = OpenAIChatModel(self.model, provider=OpenAIProvider(**provider_kwargs))
        self._agent = Agent(
            model,
            output_type=BreakdownSuggestion,
            system_prompt=SYSTEM_PROMPT,
        )
        return self._agent


def _model_name(model: str) -> str:
    """Preserve provider-specific names while accepting openai:<model> style."""

    clean = model.strip()
    if clean.startswith("openai:"):
        clean = clean.removeprefix("openai:")
    return clean


def _base_url() -> str | None:
    """Return the first configured OpenAI-compatible endpoint."""

    return (
        os.environ.get("DOGE_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("POE_BASE_URL")
        or os.environ.get("BASE_URL")
    )


class OfflineOntologist:
    """Deterministic fallback for demos and tests when no LLM credentials exist."""

    def suggest(self, rule_text: str) -> BreakdownSuggestion:
        stem = rule_text.strip().rstrip(".") or "Rule"
        return BreakdownSuggestion(
            logic_gate=LogicGate.AND,
            conditions=[
                RuleSuggestion(
                    name=f"{stem} - Explicit Mention",
                    condition=f"Text explicitly mentions the regulated concept: {stem}.",
                ),
                RuleSuggestion(
                    name=f"{stem} - Actionable Direction",
                    condition="Text pairs the concept with an instruction, recommendation, or decision.",
                ),
                RuleSuggestion(
                    name=f"{stem} - User-Specific Context",
                    condition="Text links the action to the reader's stated profile, goal, or constraints.",
                ),
            ],
        )
