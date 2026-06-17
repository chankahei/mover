"""LLM-driven extraction via pydantic-ai: shared signature, then claims.

The signature stage reads the input document, the grounding document, the
manual rules already written in logic (authoritative symbols), AND any
natural-language manual rules, so the whole vocabulary is aligned. Documents and
natural-language rules are then lowered to claims constrained to that signature.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from .ir import Claim, Formula, Signature
from .llm import build_agent, run
from .prompts import CLAIM_SYSTEM, MANUAL_SYSTEM, SIGNATURE_SYSTEM


class ExtractedClaim(BaseModel):
    """What the claim/manual agents return; wrapped into a :class:`Claim`."""

    id: str
    text: Optional[str] = None
    formula: Formula


class ExtractedClaims(BaseModel):
    claims: List[ExtractedClaim] = []


def _manual_logic_summary(manual_claims: List[Claim]) -> str:
    if not manual_claims:
        return "(none)"
    return "\n".join(
        f"- {m.text or m.id}: {m.formula.model_dump_json()}" for m in manual_claims
    )


def _manual_text_summary(manual_texts: List[str]) -> str:
    if not manual_texts:
        return "(none)"
    return "\n".join(f"- {t}" for t in manual_texts)


class Extractor:
    def signature(
        self,
        input_content: str,
        grounding_content: str,
        manual_claims: List[Claim],
        manual_texts: List[str],
    ) -> Signature:
        agent = build_agent(Signature, SIGNATURE_SYSTEM)
        user = (
            f"DOCUMENT A (generated input):\n{input_content}\n\n"
            f"DOCUMENT B (grounding source):\n{grounding_content}\n\n"
            f"MANUAL RULES IN LOGIC (authoritative symbols):\n"
            f"{_manual_logic_summary(manual_claims)}\n\n"
            f"MANUAL RULES IN NATURAL LANGUAGE:\n"
            f"{_manual_text_summary(manual_texts)}"
        )
        return run(agent, user)

    def claims(self, content: str, source: str, sig: Signature) -> List[Claim]:
        agent = build_agent(ExtractedClaims, CLAIM_SYSTEM)
        user = f"SIGNATURE:\n{sig.model_dump_json()}\n\nDOCUMENT:\n{content}"
        return self._wrap(run(agent, user), source)

    def manual_from_text(self, texts: List[str], sig: Signature) -> List[Claim]:
        if not texts:
            return []
        agent = build_agent(ExtractedClaims, MANUAL_SYSTEM)
        rules = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
        user = f"SIGNATURE:\n{sig.model_dump_json()}\n\nRULES:\n{rules}"
        return self._wrap(run(agent, user), "manual")

    @staticmethod
    def _wrap(extracted: ExtractedClaims, source: str) -> List[Claim]:
        return [
            Claim(
                id=f"{source}::{ec.id}",
                source=source,  # type: ignore[arg-type]
                text=ec.text,
                formula=ec.formula,
            )
            for ec in extracted.claims
        ]
