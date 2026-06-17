"""Build explanations for an UNSAT result.

A deterministic summary is always produced from the minimized core. When
``use_llm`` is set, an LLM is asked to paraphrase *that already-proven core*
into prose - it is never given freedom to add facts, and any failure falls back
to the deterministic summary.
"""

from __future__ import annotations

from typing import List

from .ir import Claim
from .models import Explanation
from .render import render_claim, render_formula

_LLM_SYS = (
    "You explain, in two or three plain-English sentences, why a set of logical "
    "claims is mutually contradictory. You are given the EXACT minimal set of "
    "conflicting claims that a solver already proved inconsistent. Restate only "
    "what is in that set. Do NOT introduce any new facts, numbers, or claims. "
    "Mention which claims come from the generated input, the grounding source, "
    "and the manual rules where relevant."
)


def _deterministic_summary(core: List[Claim]) -> str:
    by_source: dict[str, List[Claim]] = {}
    for c in core:
        by_source.setdefault(c.source, []).append(c)
    parts = [
        f"{len(core)} claims are mutually inconsistent",
        "(" + ", ".join(f"{len(v)} from {k}" for k, v in by_source.items()) + ").",
    ]
    parts.append(
        "Each listed claim is necessary for the contradiction: "
        + "; ".join(render_formula(c.formula) for c in core)
        + "."
    )
    return " ".join(parts)


def _llm_summary(core: List[Claim]) -> str:
    from .llm import build_agent, run

    agent = build_agent(str, _LLM_SYS)
    rendered = "\n".join(render_claim(c) for c in core)
    return run(agent, f"Minimal conflicting set:\n{rendered}")


def build_explanation(core: List[Claim], use_llm: bool) -> Explanation:
    rendered = [render_claim(c) for c in core]
    summary = _deterministic_summary(core)
    if use_llm and core:
        try:
            summary = _llm_summary(core)
        except Exception:
            pass  # fall back to the deterministic summary
    return Explanation(summary=summary, rendered_claims=rendered)
