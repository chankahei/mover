"""Compiler: lower the typed IR into Z3 expressions.

Assumes the IR has already passed :class:`~guardrail.typecheck.TypeChecker`, so
symbol lookups here are total. Builds one ``Entity`` sort, an ``EnumSort`` per
enum, and uninterpreted functions for concepts, roles, and attributes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import z3

from .ir import AttributeDecl, Signature, ValueSort
from .typecheck import IRTypeError


class Compiler:
    def __init__(self, sig: Signature, unique_names: bool = False) -> None:
        # A private context per compiler keeps datatype (enum) declarations from
        # colliding across requests in Z3's otherwise-global namespace.
        self.ctx = z3.Context()
        self.Entity = z3.DeclareSort("Entity", ctx=self.ctx)
        self.entities = {e.name: z3.Const(e.name, self.Entity) for e in sig.entities}

        self.enums: Dict[str, Tuple[z3.SortRef, Dict[str, z3.ExprRef]]] = {}
        for en in sig.enums:
            sort, consts = z3.EnumSort(en.name, en.values, ctx=self.ctx)
            self.enums[en.name] = (sort, dict(zip(en.values, consts)))

        self.concepts = {
            c.name: z3.Function(c.name, self.Entity, z3.BoolSort(self.ctx))
            for c in sig.concepts
        }
        self.roles = {
            r.name: z3.Function(r.name, self.Entity, self.Entity, z3.BoolSort(self.ctx))
            for r in sig.roles
        }
        self.attr_decls: Dict[str, AttributeDecl] = {a.name: a for a in sig.attributes}
        self.attrs = {
            a.name: z3.Function(a.name, self.Entity, self._sort(a.value_sort))
            for a in sig.attributes
        }

        # Optional unique-names assumption: all entities pairwise distinct.
        self.distinct: Optional[z3.BoolRef] = (
            z3.Distinct(*self.entities.values())
            if unique_names and len(self.entities) > 1
            else None
        )

    def solver(self) -> z3.Solver:
        """A solver bound to this compiler's context."""
        return z3.Solver(ctx=self.ctx)

    def bool_lit(self, name: str) -> z3.BoolRef:
        """A boolean literal in this compiler's context (for tracking)."""
        return z3.Bool(name, self.ctx)

    def _sort(self, vs: ValueSort) -> z3.SortRef:
        if vs.kind == "Int":
            return z3.IntSort(self.ctx)
        if vs.kind == "Real":
            return z3.RealSort(self.ctx)
        if vs.kind == "Bool":
            return z3.BoolSort(self.ctx)
        if vs.kind == "Entity":
            return self.Entity
        if vs.kind == "Enum":
            return self.enums[vs.enum][0]
        raise IRTypeError(f"unknown value sort '{vs.kind}'")

    # -- terms --
    def _term(self, t, env: Dict[str, z3.ExprRef]) -> z3.ExprRef:
        if t.node == "entity":
            return self.entities[t.name]
        return env[t.name]

    @staticmethod
    def _coerce(a: z3.ExprRef, b: z3.ExprRef) -> Tuple[z3.ExprRef, z3.ExprRef]:
        if a.sort() == b.sort():
            return a, b
        if z3.RealSort() in (a.sort(), b.sort()):
            if a.sort() == z3.IntSort():
                a = z3.ToReal(a)
            if b.sort() == z3.IntSort():
                b = z3.ToReal(b)
        return a, b

    def _num(self, t, env: Dict[str, z3.ExprRef]) -> z3.ExprRef:
        if t.node == "num":
            return z3.RealVal(t.value, self.ctx)
        if t.node == "attr":
            return self.attrs[t.attribute](self._term(t.arg, env))
        left, right = self._coerce(self._num(t.left, env), self._num(t.right, env))
        return {"+": left + right, "-": left - right, "*": left * right, "/": left / right}[t.op]

    # -- formulas --
    def formula(self, f, env: Optional[Dict[str, z3.ExprRef]] = None) -> z3.ExprRef:
        env = env or {}
        n = f.node
        if n == "bool":
            return z3.BoolVal(f.value, self.ctx)
        if n == "concept":
            return self.concepts[f.concept](self._term(f.arg, env))
        if n == "role":
            return self.roles[f.role](self._term(f.src, env), self._term(f.dst, env))
        if n == "compare":
            left, right = self._coerce(self._num(f.left, env), self._num(f.right, env))
            return {
                "<": left < right,
                "<=": left <= right,
                ">": left > right,
                ">=": left >= right,
                "==": left == right,
                "!=": left != right,
            }[f.op]
        if n == "enum_eq":
            enum = self.attr_decls[f.attribute].value_sort.enum
            val = self.enums[enum][1][f.value]
            e = self.attrs[f.attribute](self._term(f.arg, env)) == val
            return z3.Not(e) if f.negated else e
        if n == "func_eq":
            e = self.attrs[f.attribute](self._term(f.arg, env)) == self._term(f.value, env)
            return z3.Not(e) if f.negated else e
        if n == "equal":
            e = self._term(f.left, env) == self._term(f.right, env)
            return z3.Not(e) if f.negated else e
        if n == "not":
            return z3.Not(self.formula(f.arg, env))
        if n == "and":
            return z3.And([self.formula(a, env) for a in f.args])
        if n == "or":
            return z3.Or([self.formula(a, env) for a in f.args])
        if n == "implies":
            return z3.Implies(self.formula(f.hyp, env), self.formula(f.con, env))
        if n == "iff":
            return self.formula(f.left, env) == self.formula(f.right, env)
        if n == "quant":
            cs = [z3.Const(v, self.Entity) for v in f.vars]
            body = self.formula(f.body, {**env, **dict(zip(f.vars, cs))})
            return z3.ForAll(cs, body) if f.kind == "forall" else z3.Exists(cs, body)
        raise IRTypeError(f"unhandled node '{n}'")
