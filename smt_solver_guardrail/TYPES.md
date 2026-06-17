# SMT Solver Guardrail — Type Reference

Authoritative listing of every type in the system. See [`DESIGN.md`](./DESIGN.md)
for rationale. All types are Pydantic v2 models. Recursive nodes use a
`node` **discriminator** so JSON from the LLM is parsed into the right variant.

Legend for the right column: Z3 lowering / role in the IR.

---

## 1. Signature types

The shared vocabulary both documents are extracted into.

| Type | Fields | Z3 / meaning |
| --- | --- | --- |
| `ValueSort` | `kind: "Int" \| "Real" \| "Bool" \| "Enum" \| "Entity"`, `enum: str?` (required iff `kind=="Enum"`) | range of an attribute |
| `EnumDecl` | `name: str`, `values: list[str]` | `EnumSort(name, values)` — closed value domain |
| `EntityDecl` | `name: str`, `description: str?` | `Const(name, Entity)` — an individual |
| `ConceptDecl` | `name: str`, `description: str?` | `Function(name, Entity, Bool)` — unary class |
| `RoleDecl` | `name: str`, `description: str?` | `Function(name, Entity, Entity, Bool)` — binary relation |
| `AttributeDecl` | `name: str`, `value_sort: ValueSort`, `description: str?` | `Function(name, Entity, V)` — total function (functionality is free) |
| `Signature` | `enums: list[EnumDecl]`, `entities: list[EntityDecl]`, `concepts: list[ConceptDecl]`, `roles: list[RoleDecl]`, `attributes: list[AttributeDecl]` | the whole declared vocabulary |

---

## 2. Term types

`Term` — references an individual of sort `Entity`. Discriminated on `node`.

| Type | `node` | Fields | Meaning |
| --- | --- | --- | --- |
| `EntityRef` | `"entity"` | `name: str` (must be declared) | a named individual |
| `Var` | `"var"` | `name: str` (must be bound by a quantifier) | a bound variable |

```
Term = EntityRef | Var
```

---

## 3. Numeric term types

`NumTerm` — an expression of numeric sort. Discriminated on `node`.

| Type | `node` | Fields | Meaning |
| --- | --- | --- | --- |
| `NumLiteral` | `"num"` | `value: float` | numeric constant |
| `AttrApp` | `"attr"` | `attribute: str` (must be `Int`/`Real`), `arg: Term` | application of a numeric attribute, e.g. `height(Alice)` |
| `Arith` | `"arith"` | `op: "+" \| "-" \| "*" \| "/"`, `left: NumTerm`, `right: NumTerm` | arithmetic combination |

```
NumTerm = NumLiteral | AttrApp | Arith
```

Int/Real are coerced to a common sort at compile time (`ToReal` as needed).

---

## 4. Atom types (sort-segregated)

Leaf formulas. Equality is deliberately split by sort so categories cannot be
conflated.

| Type | `node` | Fields | Meaning / Z3 |
| --- | --- | --- | --- |
| `BoolConst` | `"bool"` | `value: bool` | literal `True`/`False` |
| `ConceptAssertion` | `"concept"` | `concept: str`, `arg: Term` | class membership `C(t)` |
| `RoleAssertion` | `"role"` | `role: str`, `src: Term`, `dst: Term` | relation `R(s,t)` |
| `Compare` | `"compare"` | `op: "<" \| "<=" \| ">" \| ">=" \| "==" \| "!="`, `left: NumTerm`, `right: NumTerm` | **numeric only** comparison |
| `EnumEq` | `"enum_eq"` | `attribute: str` (enum attr), `arg: Term`, `value: str` (in enum domain), `negated: bool=False` | `attr(arg) (==/!=) enumValue` |
| `FuncEq` | `"func_eq"` | `attribute: str` (entity-valued attr), `arg: Term`, `value: Term`, `negated: bool=False` | functional entity attr, e.g. `capital(France)=Paris` |
| `Equal` | `"equal"` | `left: Term`, `right: Term`, `negated: bool=False` | entity identity / distinctness |

---

## 5. Connective & quantifier types

| Type | `node` | Fields | Meaning |
| --- | --- | --- | --- |
| `Not` | `"not"` | `arg: Formula` | `¬` |
| `And` | `"and"` | `args: list[Formula]` | `∧` |
| `Or` | `"or"` | `args: list[Formula]` | `∨` |
| `Implies` | `"implies"` | `hyp: Formula`, `con: Formula` | `→` |
| `Iff` | `"iff"` | `left: Formula`, `right: Formula` | `↔` |
| `Quant` | `"quant"` | `kind: "forall" \| "exists"`, `vars: list[str]`, `body: Formula` | `∀`/`∃` over `Entity` |

```
Formula = BoolConst | ConceptAssertion | RoleAssertion | Compare
        | EnumEq | FuncEq | Equal
        | Not | And | Or | Implies | Iff | Quant
```

---

## 6. Claim (top level)

| Type | Fields | Meaning |
| --- | --- | --- |
| `Claim` | `id: str`, `source: "input" \| "grounding" \| "manual"`, `text: str?`, `formula: Formula` | one tracked, asserted fact carrying its NL span |

Each claim is asserted with `assert_and_track(compile(formula), Bool(id))` so
the unsat core maps back to claim ids.

---

## 7. Request / response types

| Type | Fields | Notes |
| --- | --- | --- |
| `ManualItem` | `Union[Claim, str]` | a manual rule: a pre-authored `Claim` **or** a natural-language string |
| `GuardrailRequest` | `input_content: str`, `grounding_content: str`, `manual_logic: list[ManualItem] = []`, `unique_names: bool = False`, `mode: "consistency" \| "entailment" = "consistency"`, `explain_with_llm: bool = True` | request body; NL manual rules are converted to `Claim`s by the LLM |
| `EntailmentResult` | `claim_id: str`, `text: str?`, `verdict: "supported" \| "unsupported" \| "contradicted"`, `status: "sat" \| "unsat" \| "unknown"`, `witness: str?` | per input-claim verdict; `witness` is the counter-model for `unsupported` |
| `Explanation` | `summary: str` (NL prose), `rendered_claims: list[str]` (deterministic renderings) | returned by default |
| `GuardrailReport` | `consistent: bool`, `status: "sat" \| "unsat" \| "unknown"`, `contradicting_claim_ids: list[str] = []`, `contradicting_claims: list[Claim] = []`, `entailment: list[EntailmentResult]? = None`, `explanation: Explanation? = None`, `model_excerpt: str? = None`, `signature: Signature`, `claims: list[Claim]` | response body |

---

## 8. Engine / helper types (non-serialized)

| Type | Role |
| --- | --- |
| `IRTypeError(Exception)` | raised by the type checker; surfaced as HTTP 422 |
| `TypeChecker` | validates a `Claim` against a `Signature` before compilation |
| `Compiler` | lowers IR → Z3 (declares `Entity`, enums, concepts, roles, attributes; compiles terms/formulas; holds optional `Distinct` axiom for UNA; one private Z3 context per instance) |
| `ExtractedClaim` | LLM output for one claim: `id: str`, `text: str?`, `formula: Formula` (wrapped into a `Claim` with its `source`) |
| `ExtractedClaims` | `claims: list[ExtractedClaim]` — the agents' structured `output_type` |
| `Extractor` | pydantic-ai-driven: `signature(input, grounding, manual_claims, manual_texts) -> Signature`; `claims(content, source, sig) -> list[Claim]`; `manual_from_text(texts, sig) -> list[Claim]` (NL rules → `manual` claims) |
| `Agent` (pydantic-ai) | typed LLM agent built by `llm.build_agent(output_type, system_prompt)`; configured from env (`OPENAI_API_KEY`/`BASE_URL`/model) |

`TypeCheckFailure` (pipeline) collects all `IRTypeError` messages and is mapped
to HTTP 422 by the API.

---

## 9. Discriminated-union summary

```
Term    = EntityRef | Var                                   # node: entity|var
NumTerm = NumLiteral | AttrApp | Arith                      # node: num|attr|arith
Formula = BoolConst | ConceptAssertion | RoleAssertion      # node: bool|concept|role
        | Compare | EnumEq | FuncEq | Equal                 #       compare|enum_eq|func_eq|equal
        | Not | And | Or | Implies | Iff | Quant            #       not|and|or|implies|iff|quant
```

`ValueSort.kind ∈ {Int, Real, Bool, Enum, Entity}` is the only other closed
enumeration in the type system.
