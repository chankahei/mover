"""Orchestration: turn a request into a report.

Stages (see DESIGN.md §4): shared signature -> per-document claims -> union with
manual logic -> type check -> compile + solve. Kept free of FastAPI so it can be
unit-tested and reused.
"""

from __future__ import annotations

from typing import List, Tuple

from .extract import Extractor
from .ir import Claim, Signature
from .models import GuardrailReport, GuardrailRequest
from .solve import solve_consistency, solve_entailment
from .typecheck import IRTypeError, TypeChecker


class TypeCheckFailure(Exception):
    """Raised when one or more claims are not well-typed; carries all errors."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def split_manual(req: GuardrailRequest) -> Tuple[List[Claim], List[str]]:
    """Partition manual_logic into pre-authored claims and NL rule strings."""
    claims: List[Claim] = []
    texts: List[str] = []
    for item in req.manual_logic:
        if isinstance(item, str):
            texts.append(item)
        else:
            claims.append(item.model_copy(update={"source": "manual"}))
    return claims, texts


def assemble_claims(
    extractor: Extractor,
    req: GuardrailRequest,
    sig: Signature,
    manual_claims: List[Claim],
    manual_texts: List[str],
) -> List[Claim]:
    return (
        extractor.claims(req.input_content, "input", sig)
        + extractor.claims(req.grounding_content, "grounding", sig)
        + extractor.manual_from_text(manual_texts, sig)
        + manual_claims
    )


def type_check_all(sig: Signature, claims: List[Claim]) -> None:
    tc = TypeChecker(sig)
    errors: List[str] = []
    for cl in claims:
        try:
            tc.check_claim(cl)
        except IRTypeError as e:
            errors.append(str(e))
    if errors:
        raise TypeCheckFailure(errors)


def solve(claims: List[Claim], sig: Signature, req: GuardrailRequest) -> GuardrailReport:
    if req.mode == "entailment":
        return solve_entailment(claims, sig, req.unique_names, req.explain_with_llm)
    return solve_consistency(claims, sig, req.unique_names, req.explain_with_llm)


def run_guardrail(req: GuardrailRequest, extractor: Extractor) -> GuardrailReport:
    manual_claims, manual_texts = split_manual(req)
    sig = extractor.signature(
        req.input_content, req.grounding_content, manual_claims, manual_texts
    )
    claims = assemble_claims(extractor, req, sig, manual_claims, manual_texts)
    type_check_all(sig, claims)
    return solve(claims, sig, req)
