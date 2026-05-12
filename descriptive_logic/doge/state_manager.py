from __future__ import annotations

import json
from pathlib import Path

from doge.models import LogicGate, OntologyNode, OntologyRulebase, RuleSuggestion, slugify_name


class OntologyState:
    """Mutable ontology tree with JSON persistence."""

    def __init__(self, path: Path, rulebase: OntologyRulebase | None = None) -> None:
        self.path = path
        self.rulebase = rulebase or OntologyRulebase(
            root=OntologyNode(
                node_id="root",
                name="Root",
                description="Root ontology node.",
                logic_gate=LogicGate.OR,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "OntologyState":
        if not path.exists():
            return cls(path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(path=path, rulebase=OntologyRulebase.from_raw_json(payload))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.rulebase.to_schema_json()
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    def find(self, node_id: str) -> OntologyNode | None:
        for node in self.rulebase.root.iter_nodes():
            if node.node_id == node_id:
                return node
        return None

    def path_to(self, node_id: str) -> list[OntologyNode]:
        path: list[OntologyNode] = []

        def visit(node: OntologyNode) -> bool:
            path.append(node)
            if node.node_id == node_id:
                return True
            for child in node.children:
                if visit(child):
                    return True
            path.pop()
            return False

        visit(self.rulebase.root)
        return path

    def parent_of(self, node_id: str) -> OntologyNode | None:
        for node in self.rulebase.root.iter_nodes():
            if any(child.node_id == node_id for child in node.children):
                return node
        return None

    def add_branch(
        self,
        parent_id: str,
        name: str,
        description: str,
        logic_gate: LogicGate,
        suggestions: list[RuleSuggestion],
    ) -> OntologyNode:
        parent = self._require_node(parent_id)
        parent.is_leaf = False
        parent.logic_gate = parent.logic_gate or LogicGate.OR
        branch_id = self._next_id(slugify_name(name))
        branch = OntologyNode(
            node_id=branch_id,
            name=name,
            description=description,
            is_leaf=False,
            logic_gate=logic_gate,
            children=self._children_from_suggestions(suggestions, reserved={branch_id}),
        )
        parent.children.append(branch)
        return branch

    def decompose_node(
        self,
        node_id: str,
        logic_gate: LogicGate,
        suggestions: list[RuleSuggestion],
    ) -> OntologyNode:
        node = self._require_node(node_id)
        node.is_leaf = False
        node.logic_gate = logic_gate
        node.children = self._children_from_suggestions(suggestions)
        return node

    def mark_leaf(self, node_id: str) -> None:
        node = self._require_node(node_id)
        node.is_leaf = True
        node.logic_gate = None
        node.children = []

    def _children_from_suggestions(
        self,
        suggestions: list[RuleSuggestion],
        reserved: set[str] | None = None,
    ) -> list[OntologyNode]:
        reserved = set(reserved or set())
        children: list[OntologyNode] = []
        for suggestion in suggestions:
            child_id = self._next_id(slugify_name(suggestion.name), reserved=reserved)
            reserved.add(child_id)
            children.append(
                OntologyNode(
                    node_id=child_id,
                    name=suggestion.name,
                    description=suggestion.condition,
                    is_leaf=True,
                )
            )
        return children

    def _require_node(self, node_id: str) -> OntologyNode:
        node = self.find(node_id)
        if node is None:
            raise KeyError(f"unknown ontology node: {node_id}")
        return node

    def _next_id(self, stem: str, reserved: set[str] | None = None) -> str:
        existing = {node.node_id for node in self.rulebase.root.iter_nodes()}
        if reserved:
            existing.update(reserved)
        clean = slugify_name(stem)
        if clean not in existing:
            return clean
        index = 2
        while f"{clean}_{index}" in existing:
            index += 1
        return f"{clean}_{index}"
