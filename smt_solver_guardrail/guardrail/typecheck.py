"""Type checker: reject malformed IR before it ever reaches Z3.

The LLM will occasionally emit claims that reference unknown symbols, put a
string where an enum value belongs, or use a non-numeric attribute in a numeric
comparison. Catching those here yields a clear HTTP 422 instead of a cryptic Z3
crash deep in compilation.
"""

from __future__ import annotations

from typing import Dict, Set

from .ir import AttributeDecl, Claim, Signature


class IRTypeError(Exception):
    """Raised when a claim is not well-typed against the signature."""


class TypeChecker:
    def __init__(self, sig: Signature) -> None:
        self.entities: Set[str] = {e.name for e in sig.entities}
        self.concepts: Set[str] = {c.name for c in sig.concepts}
        self.roles: Set[str] = {r.name for r in sig.roles}
        self.attrs: Dict[str, AttributeDecl] = {a.name: a for a in sig.attributes}
        self.enums: Dict[str, Set[str]] = {e.name: set(e.values) for e in sig.enums}

    def check_claim(self, claim: Claim) -> None:
        try:
            self._formula(claim.formula, set())
        except IRTypeError as e:
            raise IRTypeError(f"claim '{claim.id}': {e}") from e

    # -- terms --
    def _term(self, t, bound: Set[str]) -> None:
        if t.node == "entity" and t.name not in self.entities:
            raise IRTypeError(f"unknown entity '{t.name}'")
        if t.node == "var" and t.name not in bound:
            raise IRTypeError(f"unbound variable '{t.name}'")

    def _numterm(self, t, bound: Set[str]) -> None:
        if t.node == "num":
            return
        if t.node == "attr":
            a = self.attrs.get(t.attribute)
            if a is None:
                raise IRTypeError(f"unknown attribute '{t.attribute}'")
            if a.value_sort.kind not in ("Int", "Real"):
                raise IRTypeError(f"attribute '{t.attribute}' is not numeric")
            self._term(t.arg, bound)
        elif t.node == "arith":
            self._numterm(t.left, bound)
            self._numterm(t.right, bound)

    # -- formulas --
    def _formula(self, f, bound: Set[str]) -> None:
        n = f.node
        if n == "bool":
            return
        if n == "concept":
            if f.concept not in self.concepts:
                raise IRTypeError(f"unknown concept '{f.concept}'")
            self._term(f.arg, bound)
        elif n == "role":
            if f.role not in self.roles:
                raise IRTypeError(f"unknown role '{f.role}'")
            self._term(f.src, bound)
            self._term(f.dst, bound)
        elif n == "compare":
            self._numterm(f.left, bound)
            self._numterm(f.right, bound)
        elif n == "enum_eq":
            a = self.attrs.get(f.attribute)
            if a is None or a.value_sort.kind != "Enum":
                raise IRTypeError(f"'{f.attribute}' is not an enum attribute")
            dom = self.enums.get(a.value_sort.enum or "", set())
            if f.value not in dom:
                raise IRTypeError(f"'{f.value}' not in enum '{a.value_sort.enum}'")
            self._term(f.arg, bound)
        elif n == "func_eq":
            a = self.attrs.get(f.attribute)
            if a is None or a.value_sort.kind != "Entity":
                raise IRTypeError(f"'{f.attribute}' is not an entity-valued attribute")
            self._term(f.arg, bound)
            self._term(f.value, bound)
        elif n == "equal":
            self._term(f.left, bound)
            self._term(f.right, bound)
        elif n == "not":
            self._formula(f.arg, bound)
        elif n in ("and", "or"):
            for a in f.args:
                self._formula(a, bound)
        elif n == "implies":
            self._formula(f.hyp, bound)
            self._formula(f.con, bound)
        elif n == "iff":
            self._formula(f.left, bound)
            self._formula(f.right, bound)
        elif n == "quant":
            self._formula(f.body, bound | set(f.vars))
        else:  # pragma: no cover - guarded by the discriminated union
            raise IRTypeError(f"unhandled node '{n}'")
