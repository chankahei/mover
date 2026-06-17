"""Request and response models for the guardrail API.

Kept separate from ``api.py`` so the solver layer can return these without a
circular import back into the FastAPI app.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from .ir import Claim, Signature

Mode = Literal["consistency", "entailment"]
Status = Literal["sat", "unsat", "unknown"]
Verdict = Literal["supported", "unsupported", "contradicted"]

# A manual rule is either a pre-authored Claim (authoritative symbols) or a
# natural-language string that the LLM converts into a Claim, constrained to the
# shared signature.
ManualItem = Union[Claim, str]


class GuardrailRequest(BaseModel):
    input_content: str
    grounding_content: str
    manual_logic: List[ManualItem] = []
    unique_names: bool = False
    mode: Mode = "consistency"
    explain_with_llm: bool = True


class EntailmentResult(BaseModel):
    claim_id: str
    text: Optional[str] = None
    verdict: Verdict
    status: Status
    witness: Optional[str] = None  # counter-model for an unsupported claim


class Explanation(BaseModel):
    summary: str
    rendered_claims: List[str] = []


class GuardrailReport(BaseModel):
    consistent: bool
    status: Status
    contradicting_claim_ids: List[str] = []
    contradicting_claims: List[Claim] = []
    entailment: Optional[List[EntailmentResult]] = None
    explanation: Optional[Explanation] = None
    model_excerpt: Optional[str] = None
    signature: Signature
    claims: List[Claim]
