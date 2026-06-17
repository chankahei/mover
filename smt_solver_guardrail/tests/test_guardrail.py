"""Solver / type-checker tests that run fully offline (no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch

import pytest

from examples.stock_news import build_claims, build_signature
from guardrail.extract import Extractor
from guardrail.ir import AttrApp, Claim, Compare, ConceptAssertion, EntityRef, NumLiteral
from guardrail.models import GuardrailRequest
from guardrail.pipeline import run_guardrail, split_manual
from guardrail.solve import solve_consistency, solve_entailment
from guardrail.typecheck import IRTypeError, TypeChecker


def test_stock_news_is_inconsistent():
    sig, claims = build_signature(), build_claims()
    report = solve_consistency(claims, sig, una=True, explain=False)
    assert report.consistent is False
    assert report.status == "unsat"


def test_both_conflicts_surface():
    sig, claims = build_signature(), build_claims()
    report = solve_consistency(claims, sig, una=True, explain=False)
    ids = set(report.contradicting_claim_ids)
    # factual revenue conflict
    assert {"input::1", "grounding::1"} <= ids
    # policy conflict
    assert {"input::3", "manual::no_personalized_advice"} <= ids


def test_innocent_claims_excluded():
    sig, claims = build_signature(), build_claims()
    report = solve_consistency(claims, sig, una=True, explain=False)
    # growth and exchange agree with grounding and must not be blamed
    assert "input::2" not in report.contradicting_claim_ids
    assert "grounding::3" not in report.contradicting_claim_ids


def test_entailment_verdicts():
    sig, claims = build_signature(), build_claims()
    report = solve_entailment(claims, sig, una=True, explain=False)
    verdicts = {r.claim_id: r.verdict for r in report.entailment or []}
    assert verdicts["input::1"] == "contradicted"
    assert verdicts["input::2"] == "supported"
    assert verdicts["input::3"] == "contradicted"


def test_unsupported_has_witness():
    sig = build_signature()
    claims = [
        Claim(
            id="grounding::1",
            source="grounding",
            text="revenue 4.5",
            formula=Compare(
                op="==",
                left=AttrApp(attribute="revenue_q2_busd", arg=EntityRef(name="acme")),
                right=NumLiteral(value=4.5),
            ),
        ),
        Claim(
            id="input::1",
            source="input",
            text="article gives advice (not mentioned by grounding)",
            formula=ConceptAssertion(
                concept="gives_personalized_advice", arg=EntityRef(name="the_article")
            ),
        ),
    ]
    report = solve_entailment(claims, sig, una=False, explain=False)
    result = (report.entailment or [])[0]
    assert result.verdict == "unsupported"
    assert result.witness is not None


def test_consistent_when_aligned():
    sig = build_signature()
    claims = [
        Claim(
            id="input::1",
            source="input",
            text="revenue 4.5",
            formula=Compare(
                op="==",
                left=AttrApp(attribute="revenue_q2_busd", arg=EntityRef(name="acme")),
                right=NumLiteral(value=4.5),
            ),
        ),
        Claim(
            id="grounding::1",
            source="grounding",
            text="revenue 4.5",
            formula=Compare(
                op="==",
                left=AttrApp(attribute="revenue_q2_busd", arg=EntityRef(name="acme")),
                right=NumLiteral(value=4.5),
            ),
        ),
    ]
    report = solve_consistency(claims, sig, una=True, explain=False)
    assert report.consistent is True
    assert report.status == "sat"


def test_manual_logic_accepts_claim_or_string():
    req = GuardrailRequest(
        input_content="x",
        grounding_content="y",
        manual_logic=[
            "No personalized financial advice is given.",
            {
                "id": "m1",
                "source": "manual",
                "text": "t",
                "formula": {
                    "node": "concept",
                    "concept": "gives_personalized_advice",
                    "arg": {"node": "entity", "name": "the_article"},
                },
            },
        ],
    )
    manual_claims, manual_texts = split_manual(req)
    assert manual_texts == ["No personalized financial advice is given."]
    assert [c.id for c in manual_claims] == ["m1"]


def test_natural_language_manual_rule_flows_through():
    """An NL manual rule is converted and still produces the contradiction."""
    sig = build_signature()
    claims = build_claims()
    inp = [c for c in claims if c.source == "input"]
    grd = [c for c in claims if c.source == "grounding"]
    manual = [c for c in claims if c.source == "manual"]
    rule_text = "No personalized financial advice is given."

    def fake_sig(self, a, b, mc, mt):
        assert mt == [rule_text] and mc == []
        return sig

    def fake_claims(self, content, source, s):
        return inp if source == "input" else grd

    def fake_manual(self, texts, s):
        assert texts == [rule_text]
        return manual

    req = GuardrailRequest(
        input_content="...",
        grounding_content="...",
        manual_logic=[rule_text],
        unique_names=True,
        mode="consistency",
        explain_with_llm=False,
    )
    with patch.object(Extractor, "signature", fake_sig), patch.object(
        Extractor, "claims", fake_claims
    ), patch.object(Extractor, "manual_from_text", fake_manual):
        report = run_guardrail(req, Extractor())

    assert report.consistent is False
    ids = set(report.contradicting_claim_ids)
    assert {"input::3", "manual::no_personalized_advice"} <= ids


def test_typechecker_rejects_unknown_symbol():
    sig = build_signature()
    bad = Claim(
        id="x",
        source="input",
        text="bad",
        formula=ConceptAssertion(concept="not_declared", arg=EntityRef(name="acme")),
    )
    with pytest.raises(IRTypeError):
        TypeChecker(sig).check_claim(bad)


def test_typechecker_rejects_bad_enum_value():
    sig = build_signature()
    from guardrail.ir import EnumEq

    bad = Claim(
        id="x",
        source="grounding",
        text="bad enum",
        formula=EnumEq(attribute="listed_on", arg=EntityRef(name="acme"), value="LSE"),
    )
    with pytest.raises(IRTypeError):
        TypeChecker(sig).check_claim(bad)
