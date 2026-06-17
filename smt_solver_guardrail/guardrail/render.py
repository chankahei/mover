"""Deterministic structural renderer for IR formulas and claims.

This never fails and never invents facts: it is the source of truth for
explanations. The optional LLM prose layer (in ``explain.py``) is only allowed
to paraphrase what this renderer already states.
"""

from __future__ import annotations

from .ir import Claim

_CMP = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!="}


def _term(t) -> str:
    return t.name  # entity or var both render as their name


def _num(t) -> str:
    if t.node == "num":
        v = t.value
        return str(int(v)) if float(v).is_integer() else str(v)
    if t.node == "attr":
        return f"{t.attribute}({_term(t.arg)})"
    return f"({_num(t.left)} {t.op} {_num(t.right)})"


def render_formula(f) -> str:
    n = f.node
    if n == "bool":
        return "true" if f.value else "false"
    if n == "concept":
        return f"{f.concept}({_term(f.arg)})"
    if n == "role":
        return f"{f.role}({_term(f.src)}, {_term(f.dst)})"
    if n == "compare":
        return f"{_num(f.left)} {_CMP[f.op]} {_num(f.right)}"
    if n == "enum_eq":
        op = "!=" if f.negated else "=="
        return f"{f.attribute}({_term(f.arg)}) {op} {f.value}"
    if n == "func_eq":
        op = "!=" if f.negated else "=="
        return f"{f.attribute}({_term(f.arg)}) {op} {_term(f.value)}"
    if n == "equal":
        op = "!=" if f.negated else "=="
        return f"{_term(f.left)} {op} {_term(f.right)}"
    if n == "not":
        return f"\u00ac {render_formula(f.arg)}"
    if n == "and":
        return "(" + " \u2227 ".join(render_formula(a) for a in f.args) + ")"
    if n == "or":
        return "(" + " \u2228 ".join(render_formula(a) for a in f.args) + ")"
    if n == "implies":
        return f"({render_formula(f.hyp)} \u2192 {render_formula(f.con)})"
    if n == "iff":
        return f"({render_formula(f.left)} \u2194 {render_formula(f.right)})"
    if n == "quant":
        q = "\u2200" if f.kind == "forall" else "\u2203"
        return f"{q}{','.join(f.vars)}. {render_formula(f.body)}"
    return f"<{n}>"


def render_claim(claim: Claim) -> str:
    """A one-line, audit-friendly rendering with id, source, text and logic."""
    text = f' "{claim.text}"' if claim.text else ""
    return f"[{claim.id}] ({claim.source}){text} \u27e6 {render_formula(claim.formula)} \u27e7"
