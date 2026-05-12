from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class LogicGate(str, Enum):
    """Boolean rule used to determine whether a parent node is triggered."""

    AND = "AND"
    OR = "OR"


class RuleSuggestion(BaseModel):
    """One LLM-proposed predicate for a parent rule."""

    model_config = ConfigDict(validate_by_name=True)

    name: str = Field(..., min_length=1, validation_alias=AliasChoices("name", "predicate"))
    condition: str = Field(..., min_length=1, validation_alias=AliasChoices("condition", "description"))


class BreakdownSuggestion(BaseModel):
    """Structured response expected from the ontologist agent."""

    logic_gate: LogicGate = LogicGate.AND
    conditions: list[RuleSuggestion] = Field(default_factory=list)


class OntologyNode(BaseModel):
    """A node in the ontology DAG serialized to ontology_rules.json."""

    node_id: str
    name: str
    description: str = ""
    is_leaf: bool = False
    logic_gate: LogicGate | None = None
    children: list["OntologyNode"] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_leaf_state(self) -> "OntologyNode":
        if self.is_leaf:
            self.children = []
            self.logic_gate = None
        elif self.logic_gate is None:
            self.logic_gate = LogicGate.OR
        return self

    def iter_nodes(self) -> list["OntologyNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.iter_nodes())
        return nodes

    def leaf_nodes(self) -> list["OntologyNode"]:
        return [node for node in self.iter_nodes() if node.is_leaf]


class ComplianceRule(BaseModel):
    """Extra compliance constraint layered onto the ontology.

    Supported kinds:
    - implication: if all antecedents are true, consequent must be true.
    - mutex: not all listed terms may be true at once.
    - requires: all listed terms must be true.
    - forbids: all listed terms must be false.
    """

    rule_id: str
    description: str = ""
    kind: Literal["implication", "mutex", "requires", "forbids"]
    antecedents: list[str] = Field(default_factory=list)
    consequent: str | None = None
    terms: list[str] = Field(default_factory=list)


class OntologyRulebase(BaseModel):
    """Persisted ontology file, with optional solver-only constraints."""

    root: OntologyNode
    constraints: list[ComplianceRule] = Field(default_factory=list)

    @classmethod
    def from_raw_json(cls, payload: dict) -> "OntologyRulebase":
        if "root" in payload:
            return cls.model_validate(payload)
        return cls(root=OntologyNode.model_validate(payload))

    def to_schema_json(self) -> dict:
        payload = self.root.model_dump(mode="json", exclude_none=True)
        if self.constraints:
            payload = {"root": payload, "constraints": [c.model_dump() for c in self.constraints]}
        return payload


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return slug or "Rule"
