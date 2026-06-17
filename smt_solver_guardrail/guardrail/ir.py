"""The typed intermediate representation (IR).

This is the contract between natural language and the Z3 solver: a strongly
typed, sort-segregated description-logic fragment. Everything else in the
service either produces (extraction) or consumes (type check, compile) these
models. See ``DESIGN.md`` and ``TYPES.md`` for the rationale.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Signature: the shared vocabulary both documents are extracted into.
# ---------------------------------------------------------------------------


class ValueSort(BaseModel):
    """The range of an attribute (a function ``Entity -> V``)."""

    kind: Literal["Int", "Real", "Bool", "Enum", "Entity"]
    enum: Optional[str] = None  # required iff kind == "Enum"


class EnumDecl(BaseModel):
    """A closed, finite value domain compiled to a Z3 ``EnumSort``."""

    name: str
    values: List[str]


class EntityDecl(BaseModel):
    """A named individual of the single uninterpreted sort ``Entity``."""

    name: str
    description: Optional[str] = None


class ConceptDecl(BaseModel):
    """A unary predicate ``Entity -> Bool`` (a class)."""

    name: str
    description: Optional[str] = None


class RoleDecl(BaseModel):
    """A binary predicate ``Entity x Entity -> Bool`` (a relation)."""

    name: str
    description: Optional[str] = None


class AttributeDecl(BaseModel):
    """A total function ``Entity -> ValueSort``. Functionality is free."""

    name: str
    value_sort: ValueSort
    description: Optional[str] = None


class Signature(BaseModel):
    """The whole declared vocabulary shared by every claim."""

    enums: List[EnumDecl] = []
    entities: List[EntityDecl] = []
    concepts: List[ConceptDecl] = []
    roles: List[RoleDecl] = []
    attributes: List[AttributeDecl] = []


# ---------------------------------------------------------------------------
# Terms: references to individuals of sort Entity.
# ---------------------------------------------------------------------------


class EntityRef(BaseModel):
    node: Literal["entity"] = "entity"
    name: str


class Var(BaseModel):
    node: Literal["var"] = "var"
    name: str


Term = Annotated[Union[EntityRef, Var], Field(discriminator="node")]


# ---------------------------------------------------------------------------
# Numeric terms.
# ---------------------------------------------------------------------------


class NumLiteral(BaseModel):
    node: Literal["num"] = "num"
    value: float


class AttrApp(BaseModel):
    """Application of a numeric attribute, e.g. ``height(Alice)``."""

    node: Literal["attr"] = "attr"
    attribute: str
    arg: Term


class Arith(BaseModel):
    node: Literal["arith"] = "arith"
    op: Literal["+", "-", "*", "/"]
    left: "NumTerm"
    right: "NumTerm"


NumTerm = Annotated[Union[NumLiteral, AttrApp, Arith], Field(discriminator="node")]


# ---------------------------------------------------------------------------
# Atoms: leaf formulas, with equality segregated by sort.
# ---------------------------------------------------------------------------


class BoolConst(BaseModel):
    node: Literal["bool"] = "bool"
    value: bool


class ConceptAssertion(BaseModel):
    node: Literal["concept"] = "concept"
    concept: str
    arg: Term


class RoleAssertion(BaseModel):
    node: Literal["role"] = "role"
    role: str
    src: Term
    dst: Term


class Compare(BaseModel):
    """Numeric-only comparison."""

    node: Literal["compare"] = "compare"
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    left: NumTerm
    right: NumTerm


class EnumEq(BaseModel):
    """``attribute(arg) (==|!=) value`` over a closed enum domain."""

    node: Literal["enum_eq"] = "enum_eq"
    attribute: str
    arg: Term
    value: str
    negated: bool = False


class FuncEq(BaseModel):
    """Entity-valued functional attribute, e.g. ``capital(France)=Paris``."""

    node: Literal["func_eq"] = "func_eq"
    attribute: str
    arg: Term
    value: Term
    negated: bool = False


class Equal(BaseModel):
    """Entity identity / distinctness."""

    node: Literal["equal"] = "equal"
    left: Term
    right: Term
    negated: bool = False


# ---------------------------------------------------------------------------
# Connectives and quantifiers.
# ---------------------------------------------------------------------------


class Not(BaseModel):
    node: Literal["not"] = "not"
    arg: "Formula"


class And(BaseModel):
    node: Literal["and"] = "and"
    args: List["Formula"]


class Or(BaseModel):
    node: Literal["or"] = "or"
    args: List["Formula"]


class Implies(BaseModel):
    node: Literal["implies"] = "implies"
    hyp: "Formula"
    con: "Formula"


class Iff(BaseModel):
    node: Literal["iff"] = "iff"
    left: "Formula"
    right: "Formula"


class Quant(BaseModel):
    node: Literal["quant"] = "quant"
    kind: Literal["forall", "exists"]
    vars: List[str]  # variables of sort Entity
    body: "Formula"


Formula = Annotated[
    Union[
        BoolConst,
        ConceptAssertion,
        RoleAssertion,
        Compare,
        EnumEq,
        FuncEq,
        Equal,
        Not,
        And,
        Or,
        Implies,
        Iff,
        Quant,
    ],
    Field(discriminator="node"),
]


# ---------------------------------------------------------------------------
# Top-level claim.
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """One tracked, asserted fact carrying its natural-language span."""

    id: str
    source: Literal["input", "grounding", "manual"]
    text: Optional[str] = None
    formula: Formula


# Resolve recursive forward references.
for _m in (Arith, Not, And, Or, Implies, Iff, Quant, Claim):
    _m.model_rebuild()
