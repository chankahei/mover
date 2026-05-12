from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doge.models import ComplianceRule, LogicGate, OntologyNode, OntologyRulebase, slugify_name


@dataclass(frozen=True)
class VerificationReport:
    status: str
    leaf_count: int
    hierarchy_constraint_count: int
    compliance_constraint_count: int
    unsat_core: list[str]

    @property
    def is_satisfiable(self) -> bool:
        return self.status == "sat"


class Z3OntologyValidator:
    """Translate an ontology rulebase into Z3 boolean constraints."""

    def __init__(self, rulebase: OntologyRulebase) -> None:
        self.rulebase = rulebase
        self._vars: dict[str, Any] = {}
        self._aliases: dict[str, str] = {}

    def verify(self) -> VerificationReport:
        try:
            import z3
        except ImportError as exc:
            raise RuntimeError("z3-solver is required. Install this package with `pip install -e .`.") from exc

        solver = z3.Solver()
        self._index_nodes(self.rulebase.root)
        hierarchy_count = self._add_hierarchy_constraints(solver, z3, self.rulebase.root)
        compliance_count = 0
        for rule in self.rulebase.constraints:
            self._add_compliance_constraint(solver, z3, rule)
            compliance_count += 1

        result = solver.check()
        core = [str(item) for item in solver.unsat_core()] if result == z3.unsat else []
        return VerificationReport(
            status=str(result),
            leaf_count=len(self.rulebase.root.leaf_nodes()),
            hierarchy_constraint_count=hierarchy_count,
            compliance_constraint_count=compliance_count,
            unsat_core=core,
        )

    def _index_nodes(self, root: OntologyNode) -> None:
        for node in root.iter_nodes():
            self._var(node.node_id)
            self._aliases[node.node_id] = node.node_id
            self._aliases[node.name] = node.node_id
            self._aliases[slugify_name(node.name)] = node.node_id

    def _add_hierarchy_constraints(self, solver: Any, z3: Any, node: OntologyNode) -> int:
        count = 0
        if node.children:
            child_vars = [self._var(child.node_id) for child in node.children]
            if node.logic_gate == LogicGate.AND:
                expression = self._var(node.node_id) == z3.And(*child_vars)
            else:
                expression = self._var(node.node_id) == z3.Or(*child_vars)
            solver.assert_and_track(expression, f"hierarchy:{node.node_id}")
            count += 1
        for child in node.children:
            count += self._add_hierarchy_constraints(solver, z3, child)
        return count

    def _add_compliance_constraint(self, solver: Any, z3: Any, rule: ComplianceRule) -> None:
        if rule.kind == "implication":
            if not rule.antecedents or not rule.consequent:
                raise ValueError(f"{rule.rule_id} requires antecedents and consequent")
            expr = z3.Implies(
                z3.And(*[self._resolve(term) for term in rule.antecedents]),
                self._resolve(rule.consequent),
            )
        elif rule.kind == "mutex":
            if len(rule.terms) < 2:
                raise ValueError(f"{rule.rule_id} mutex requires at least two terms")
            expr = z3.Not(z3.And(*[self._resolve(term) for term in rule.terms]))
        elif rule.kind == "requires":
            if not rule.terms:
                raise ValueError(f"{rule.rule_id} requires at least one term")
            expr = z3.And(*[self._resolve(term) for term in rule.terms])
        elif rule.kind == "forbids":
            if not rule.terms:
                raise ValueError(f"{rule.rule_id} requires at least one term")
            expr = z3.And(*[z3.Not(self._resolve(term)) for term in rule.terms])
        else:
            raise ValueError(f"unsupported compliance rule kind: {rule.kind}")
        solver.assert_and_track(expr, rule.rule_id)

    def _resolve(self, term: str) -> Any:
        return self._var(self._aliases.get(term, slugify_name(term)))

    def _var(self, key: str) -> Any:
        try:
            import z3
        except ImportError as exc:
            raise RuntimeError("z3-solver is required. Install this package with `pip install -e .`.") from exc

        if key not in self._vars:
            self._vars[key] = z3.Bool(key)
        return self._vars[key]


def load_rulebase(path: Path) -> OntologyRulebase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OntologyRulebase.from_raw_json(payload)


def verify_rulebase(path: Path) -> VerificationReport:
    return Z3OntologyValidator(load_rulebase(path)).verify()
