"""The DESIGN.md stock-news example, runnable WITHOUT any LLM.

It hand-builds the shared signature and claims (the output the extractor would
produce) and runs the solver directly, so you can see the contradiction
reporting end-to-end offline:

    python -m examples.stock_news
"""

from __future__ import annotations

from guardrail.ir import (
    AttributeDecl,
    AttrApp,
    Claim,
    Compare,
    ConceptAssertion,
    ConceptDecl,
    EntityDecl,
    EntityRef,
    EnumDecl,
    EnumEq,
    Not,
    NumLiteral,
    Signature,
    ValueSort,
)
from guardrail.solve import solve_consistency, solve_entailment


def build_signature() -> Signature:
    return Signature(
        enums=[EnumDecl(name="Exchange", values=["NYSE", "NASDAQ"])],
        entities=[
            EntityDecl(name="acme", description="Acme Corp, ticker ACME"),
            EntityDecl(name="the_article", description="the generated stock news article"),
        ],
        concepts=[
            ConceptDecl(
                name="gives_personalized_advice",
                description="article tells the reader to take a personal investment action",
            )
        ],
        attributes=[
            AttributeDecl(name="revenue_q2_busd", value_sort=ValueSort(kind="Real")),
            AttributeDecl(name="yoy_growth_pct", value_sort=ValueSort(kind="Real")),
            AttributeDecl(
                name="listed_on", value_sort=ValueSort(kind="Enum", enum="Exchange")
            ),
        ],
    )


def _rev(value: float):
    return Compare(
        op="==",
        left=AttrApp(attribute="revenue_q2_busd", arg=EntityRef(name="acme")),
        right=NumLiteral(value=value),
    )


def _growth(value: float):
    return Compare(
        op="==",
        left=AttrApp(attribute="yoy_growth_pct", arg=EntityRef(name="acme")),
        right=NumLiteral(value=value),
    )


def build_claims() -> list[Claim]:
    advice = ConceptAssertion(
        concept="gives_personalized_advice", arg=EntityRef(name="the_article")
    )
    return [
        Claim(id="input::1", source="input", text="revenue of $5.0 billion", formula=_rev(5.0)),
        Claim(id="input::2", source="input", text="up 12% year over year", formula=_growth(12.0)),
        Claim(
            id="input::3",
            source="input",
            text="you should buy ACME shares for your retirement portfolio",
            formula=advice,
        ),
        Claim(id="grounding::1", source="grounding", text="Q2 revenue was $4.5 billion", formula=_rev(4.5)),
        Claim(id="grounding::2", source="grounding", text="a 12% increase YoY", formula=_growth(12.0)),
        Claim(
            id="grounding::3",
            source="grounding",
            text="listed on the NYSE",
            formula=EnumEq(attribute="listed_on", arg=EntityRef(name="acme"), value="NYSE"),
        ),
        Claim(
            id="manual::no_personalized_advice",
            source="manual",
            text="The article must not give personalized financial advice.",
            formula=Not(arg=advice),
        ),
    ]


def main() -> None:
    sig = build_signature()
    claims = build_claims()

    print("=== consistency mode ===")
    report = solve_consistency(claims, sig, una=True, explain=False)
    print("consistent:", report.consistent, "| status:", report.status)
    print("conflicting claim ids:", report.contradicting_claim_ids)
    if report.explanation:
        print("\nexplanation:\n", report.explanation.summary)
        print("\nminimal conflicting set:")
        for line in report.explanation.rendered_claims:
            print("  -", line)

    print("\n=== entailment mode ===")
    ent = solve_entailment(claims, sig, una=True, explain=False)
    for r in ent.entailment or []:
        print(f"  {r.claim_id:>10}: {r.verdict}")


if __name__ == "__main__":
    main()
