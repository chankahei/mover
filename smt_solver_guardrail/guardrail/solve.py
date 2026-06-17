"""Solvers: consistency (with minimal UNSAT cores) and entailment.

Consistency uses assumption literals (one per claim) so the unsat core maps
back to claim ids. Cores are then shrunk by deletion to a locally minimal set,
and we iterate to surface *multiple independent* contradictions rather than
only the first one Z3 returns.
"""

from __future__ import annotations

from typing import List, Tuple

import z3

from .compile import Compiler
from .explain import build_explanation
from .ir import Claim, Signature
from .models import EntailmentResult, Explanation, GuardrailReport

_UNA_ID = "__unique_names__"


def _is_unsat(solver: z3.Solver, assumptions: List[z3.BoolRef]) -> bool:
    return solver.check(assumptions) == z3.unsat


def _minimize(solver: z3.Solver, core: List[z3.BoolRef]) -> List[z3.BoolRef]:
    """Deletion-based MUS shrink: drop redundant assumptions."""
    minimal = list(core)
    i = 0
    while i < len(minimal):
        candidate = minimal[:i] + minimal[i + 1 :]
        if candidate and _is_unsat(solver, candidate):
            minimal = candidate  # the dropped literal was not needed
        else:
            i += 1
    return minimal


def _collect_cores(
    solver: z3.Solver, lits: List[z3.BoolRef]
) -> List[List[z3.BoolRef]]:
    """Find several locally-minimal cores covering independent conflicts."""
    cores: List[List[z3.BoolRef]] = []
    active = list(lits)
    while _is_unsat(solver, active):
        raw = list(solver.unsat_core())
        present = {str(b) for b in raw}
        core = _minimize(solver, [a for a in active if str(a) in present] or active)
        cores.append(core)
        # Break this conflict by removing one of its members, then look again.
        drop = core[0]
        active = [a for a in active if a is not drop]
        if not active:
            break
    return cores


def solve_consistency(
    claims: List[Claim], sig: Signature, una: bool, explain: bool
) -> GuardrailReport:
    c = Compiler(sig, una)
    s = c.solver()
    lits = {cl.id: c.bool_lit(cl.id) for cl in claims}
    by_id = {cl.id: cl for cl in claims}

    for cl in claims:
        s.add(z3.Implies(lits[cl.id], c.formula(cl.formula)))
    assumptions = list(lits.values())
    if c.distinct is not None:
        una_lit = c.bool_lit(_UNA_ID)
        s.add(z3.Implies(una_lit, c.distinct))
        assumptions.append(una_lit)

    r = s.check(assumptions)
    if r == z3.sat:
        return GuardrailReport(
            consistent=True,
            status="sat",
            model_excerpt=str(s.model())[:2000],
            signature=sig,
            claims=claims,
        )
    if r == z3.unknown:
        return GuardrailReport(
            consistent=False, status="unknown", signature=sig, claims=claims
        )

    cores = _collect_cores(s, assumptions)
    core_ids: List[str] = []
    for core in cores:
        for lit in core:
            name = str(lit)
            if name not in core_ids:
                core_ids.append(name)

    offending = [by_id[i] for i in core_ids if i in by_id]
    explanation: Explanation | None = None
    if offending:
        explanation = build_explanation(offending, use_llm=explain)

    return GuardrailReport(
        consistent=False,
        status="unsat",
        contradicting_claim_ids=core_ids,
        contradicting_claims=offending,
        explanation=explanation,
        signature=sig,
        claims=claims,
    )


def _entailment_verdict(
    c: Compiler, base: List[Claim], target: Claim, una: bool
) -> Tuple[EntailmentResult, None]:
    distinct = [c.distinct] if c.distinct is not None else []

    # 1. contradicted?  base ∧ claim  UNSAT
    s = c.solver()
    for b in base:
        s.add(c.formula(b.formula))
    for d in distinct:
        s.add(d)
    s.add(c.formula(target.formula))
    if s.check() == z3.unsat:
        return EntailmentResult(
            claim_id=target.id, text=target.text, verdict="contradicted", status="unsat"
        ), None

    # 2. supported?  base ∧ ¬claim  UNSAT
    s = c.solver()
    for b in base:
        s.add(c.formula(b.formula))
    for d in distinct:
        s.add(d)
    s.add(z3.Not(c.formula(target.formula)))
    r = s.check()
    if r == z3.unsat:
        return EntailmentResult(
            claim_id=target.id, text=target.text, verdict="supported", status="unsat"
        ), None

    # 3. otherwise unsupported, with a counter-model witness
    witness = str(s.model())[:2000] if r == z3.sat else None
    return EntailmentResult(
        claim_id=target.id,
        text=target.text,
        verdict="unsupported",
        status="sat" if r == z3.sat else "unknown",
        witness=witness,
    ), None


def solve_entailment(
    claims: List[Claim], sig: Signature, una: bool, explain: bool
) -> GuardrailReport:
    c = Compiler(sig, una)
    base = [cl for cl in claims if cl.source in ("grounding", "manual")]
    targets = [cl for cl in claims if cl.source == "input"]

    results: List[EntailmentResult] = []
    for tgt in targets:
        res, _ = _entailment_verdict(c, base, tgt, una)
        results.append(res)

    all_ok = all(r.verdict == "supported" for r in results)
    return GuardrailReport(
        consistent=all_ok,
        status="unsat" if all_ok else "sat",
        entailment=results,
        signature=sig,
        claims=claims,
    )
