# SMT Solver Guardrail — Design Document

A FastAPI service that checks whether LLM-generated content is logically
consistent with its grounding (source) material, by compiling natural-language
claims into a strongly-typed description-logic IR and discharging them with the
Z3 SMT solver. When a contradiction exists, the service returns a **minimal**
set of conflicting claims plus a **natural-language explanation by default**.

> Companion file: [`TYPES.md`](./TYPES.md) is the authoritative listing of every
> type in the IR, request, and response models.

---

## 1. Problem & Scope

We want a guardrail that, given some generated `input_content`, the
`grounding_content` it was supposed to be faithful to, and optional
`manual_logic` (hand-authored business rules — each rule is **either** a
pre-authored `Claim` in IR form **or** a natural-language string that the
service converts to a `Claim` via the LLM), answers:

- Is the input **logically consistent** with the grounding + rules?
- If not, **which specific sentences conflict**, and **why**?

The service catches **logical contradictions** (and, in the stronger mode,
**lack of entailment**). It does **not** judge claims whose truth depends on
world knowledge the grounding does not encode.

### Two semantic modes

| Mode | Question answered | Catches |
| --- | --- | --- |
| `consistency` (default) | Is `input ∧ grounding ∧ manual` satisfiable? | Direct contradictions |
| `entailment` | For each input claim `c`, does `grounding ∧ manual ⊨ c`? | Contradicted **and** unsupported (hallucinated) claims |

`consistency` mode passes any input that does not actively conflict — including
a hallucination that is simply *not mentioned* by the grounding. `entailment`
mode is the stronger guardrail: it demands positive support for every input
claim.

---

## 2. Why the IR is the hard part

The FastAPI plumbing and the Z3 call are trivial. All the difficulty lives in
the intermediate representation (IR) between natural language and the solver.
Two non-obvious requirements drive the entire design.

### 2.1 Shared signature

A contradiction between input and grounding is only **detectable if both
extractions use the same symbols**. If the input says *"the CEO is John"* and
the grounding says *"the chief executive is Jane,"* the LLM must normalize both
to one symbol `ceo_of(company)` over canonical entities — otherwise the two
claims live in disjoint vocabularies and never collide.

Extraction is therefore a **two-stage pipeline**:

1. **Signature extraction** — the LLM reads *both* documents **and the
   `manual_logic`** (both the pre-authored IR claims and any natural-language
   rules) jointly and emits a single shared `Signature` (canonical entities,
   concepts, roles, attributes, enums), normalizing synonyms and coreferences.
   Manual rules already written in IR form anchor the signature: the LLM must
   include and align with their symbols so a rule can actually collide with
   extracted claims. Natural-language rules are treated like another document
   for the purpose of building the vocabulary.
2. **Claim extraction** — the LLM extracts claims from each document
   *constrained to that signature*, forbidden from inventing new symbols. The
   same step converts natural-language manual rules into `Claim`s
   (`source="manual"`) using a policy-aware prompt (a prohibition becomes a
   negated atom, a requirement a positive one).

### 2.2 Sort discipline

An unconstrained LLM will happily emit `height(Alice) = "tall"` or
`parent(France, blue)`. Z3 either rejects this cryptically or silently models
nonsense. The IR is therefore **strongly typed**, and a **type checker runs
before compilation** to reject malformed output with a clear error instead of a
Z3 crash.

Crucially, **equality is split by sort** so the LLM cannot conflate categories.
Instead of one overloaded `=`, the IR has distinct atom kinds:

- `equal` — entity identity (`Paris = the_capital`)
- `compare` — numeric order/equality (`height(Alice) > 180`)
- `enum_eq` — attribute equals a closed enum value (`color(sky) = blue`)
- `func_eq` — entity-valued functional attribute (`capital(France) = Paris`)
- `concept` / `role` — class membership and relations

---

## 3. The Type System (overview)

The universe is a **single uninterpreted sort `Entity`** for all individuals.
This is closer to description-logic style and keeps cross-references and
equality cheap. On top of `Entity`:

| Construct | Z3 shape | Meaning | Free property it buys |
| --- | --- | --- | --- |
| **Concept** | `Entity → Bool` | unary class (e.g. `Mammal`) | — |
| **Role** | `Entity × Entity → Bool` | binary relation (e.g. `parent_of`) | — |
| **Attribute** | `Entity → V` | function to a value sort | **functionality is free** |
| **Enum** | `EnumSort` | closed finite value domain | clean closed-set contradictions |

`V ∈ {Int, Real, Bool, Enum<E>, Entity}`.

Because attributes are **functions**, functionality comes for free:
`height(Alice)=180 ∧ height(Alice)=190` is automatically UNSAT with no extra
axioms. Likewise enums compiled to `EnumSort` make *"the sky is blue"* vs.
*"the sky is red"* a clean contradiction over a closed set.

Formulas are a standard typed first-order fragment: `¬, ∧, ∨, →, ↔`, plus
`∀`/`∃` over `Entity`. The atoms are the sort-segregated kinds listed in §2.2.

### 3.1 Unique-names assumption (configurable)

Whether distinct entity names denote distinct individuals is a **per-request
flag** (`unique_names`, default `false`).

- **UNA off:** `Paris` and `Lyon` might be the same entity, so
  `capital(France)=Paris ∧ capital(France)=Lyon` is *satisfiable*. The LLM is
  instructed to emit explicit (in)equalities (`equal` with `negated=true`) when
  two things differ.
- **UNA on:** all declared entities are pairwise `Distinct`, so the above is a
  contradiction.

### 3.2 Claim tracking

Every top-level `Claim` carries `id`, `source` (`input` / `grounding` /
`manual`), and the original NL `text`. Each claim is asserted under a Z3
tracking literal named by its `id` (`assert_and_track`), so `unsat_core()`
returns exactly the claim ids that jointly contradict — which we resolve back to
full `Claim` objects (with their source spans).

The full type listing is in [`TYPES.md`](./TYPES.md).

---

## 4. Pipeline

```
                 input_content   grounding_content   manual_logic
                       │                │                  │
        ┌──────────────┴────────────────┴──────────────────┤
        ▼                                                   │
  (1) Signature extraction  ── LLM reads BOTH docs AND      │
        │                      manual_logic (IR + NL) ──► shared Signature
        ▼                                                   │
  (2) Claim extraction      ── LLM, constrained to Signature│
        │   input claims + grounding claims + NL→manual     │
        └───────────────┬───────────────────────────────────┘
                        ▼
              (3) Union of all claims (input ∪ grounding ∪ manual)
                        ▼
              (4) Type check  ── reject malformed IR (HTTP 422)
                        ▼
              (5) Compile IR → Z3
                        ▼
              (6) Solve:  consistency  |  entailment
                        ▼
              (7) Minimize core + render explanation
                        ▼
                   GuardrailReport (with explanation by default)
```

Stages:

1. **Signature** — one shared `Signature` built from *all* inputs: both
   documents **and** the `manual_logic`. Manual items may be pre-authored
   `Claim`s (their symbols are authoritative and must appear in the signature)
   or natural-language strings (treated like another document). Aligning the
   document phrasing onto these canonical symbols is what lets a manual rule
   actually collide with an extracted claim.
2. **Claims** — per-document extraction constrained to the signature, plus
   conversion of any natural-language manual rules into `manual` claims.
3. **Union** — combine input, grounding, NL-derived manual, and pre-authored
   manual claims.
4. **Type check** — `TypeChecker` validates every claim against the signature
   before anything reaches Z3; failures return HTTP 422 with the offending
   claim ids and messages.
5. **Compile** — `Compiler` lowers the IR to Z3 sorts/functions/formulas,
   declaring `Entity`, enums (`EnumSort`), concepts, roles, and attributes.
6. **Solve** — see §5.
7. **Explain** — see §6.

---

## 5. Solving

### 5.1 Consistency (UNSAT core)

Use assumption-based checking: assert each claim with `assert_and_track(formula,
Bool(claim_id))` (plus an optional `__unique_names__` literal guarding the
`Distinct` axiom), then `check()`.

- `sat` → consistent; return a model excerpt as a witness world.
- `unsat` → inconsistent; take `unsat_core()`, **minimize it** (§6.1), and
  report the conflicting claim ids + resolved `Claim` objects.
- `unknown` → reported honestly (e.g. quantifiers pushed Z3 past decidability)
  rather than pretending it is `True`.

### 5.2 Entailment (three-way verdict)

Split claims into `base = grounding ∪ manual` and `targets = input`. For each
target claim `c`, "not entailed" hides two very different verdicts a reviewer
cares about, so we return a **three-way** result:

1. `base ∧ c` is **UNSAT** → **contradicted**.
2. else `base ∧ ¬c` is **UNSAT** → **supported** (grounding entails `c`).
3. else → **unsupported** — and we hand back a **counter-model witness**: a
   concrete world in which the grounding holds but the claim is false. That
   witness is the precise evidence the claim was unfounded.

---

## 6. Explanation (returned by default)

Turning "here are the claim ids in the core" into a real explanation needs two
things: the core must be **minimal**, and it must be **rendered back to
language**.

### 6.1 Minimal core via deletion (MUS shrink)

Z3's raw `unsat_core()` can include irrelevant claims. We run a
**deletion-based MUS loop**: drop each core literal and re-check; if the set is
still UNSAT without it, that literal was redundant and is removed. The result is
a **locally minimal** core where *every* claim is necessary for the
contradiction — so the explanation never blames an innocent sentence.

- Cost: `O(n)` extra solver calls.
- Guarantee: locally minimal (every member necessary). Not guaranteed to be the
  *globally smallest* core, but the "every claim here is genuinely part of the
  conflict" property is the one that matters for guardrail reporting.

### 6.2 Deterministic renderer + optional LLM prose

- A **structural renderer** always runs and never fails, producing a faithful
  rendering of each formula (e.g. `⟦ color(sky) = red ⟧`). This is the **source
  of truth**.
- An **LLM** optionally turns the minimal core into natural prose. It is
  **confined to paraphrasing an already-proven minimal set** — never to adding
  facts. If the LLM is dropped or fails, you still get a correct (terser)
  explanation rather than a fabrication.

### 6.3 Example explanation

For input claiming the sky is red and grounding saying it is blue, the minimized
core has exactly two claims:

> The input states the sky is red while the source states it is blue; a single
> object cannot have two different colors from a closed color set.
>
> Minimal conflicting set:
> - `[input::2]` (input) "The sky was red that evening." ⟦ color(sky) = red ⟧
> - `[grounding::5]` (grounding) "The sky was a clear blue." ⟦ color(sky) = blue ⟧

---

## 7. API

### Endpoint

`POST /guardrail` → `GuardrailReport`

Request (`GuardrailRequest`):

```json
{
  "input_content": "…generated text…",
  "grounding_content": "…source text…",
  "manual_logic": [
    "No personalized financial advice is given.",
    { "id": "r1", "source": "manual", "text": "…", "formula": { "node": "…" } }
  ],
  "unique_names": false,
  "mode": "consistency",
  "explain_with_llm": true
}
```

Each `manual_logic` entry is **either** a natural-language string (converted to
a `Claim` by the LLM, constrained to the shared signature) **or** a pre-authored
`Claim` object. The two forms can be mixed in one request.

Response is a `GuardrailReport` (see [`TYPES.md`](./TYPES.md)) carrying the
verdict, the minimized conflicting claims, the entailment verdicts (in
entailment mode), and the explanation by default.

### LLM access

All LLM calls go through **pydantic-ai** typed agents (`guardrail/llm.py`): each
agent declares a Pydantic `output_type` (`Signature`, the extracted-claims
model, or `str` for explanation prose) and pydantic-ai handles structured
decoding and validation. The model is configured from the environment
(`OPENAI_API_KEY`, optional `BASE_URL`, and `GUARDRAIL_LLM_MODEL` /
`DOGE_LLM_MODEL`), so any OpenAI-compatible gateway works.

---

## 8. Design boundaries & caveats

- **Logical contradictions only.** Cannot judge claims whose truth needs world
  knowledge the grounding does not encode.
- **Consistency ≠ support.** `consistency` mode passes anything that does not
  actively conflict; use `entailment` mode for "is this claim supported?".
- **Decidability.** Quantifiers over the uninterpreted `Entity` sort can push Z3
  to `unknown`. We keep the fragment effectively decidable by leaning on
  functional attributes and enums instead of heavy nested quantification, and we
  surface `unknown` honestly.
- **Extraction fidelity is the trust root.** The type checker guarantees the IR
  is well-formed and won't crash Z3, but it cannot guarantee the extraction
  matched the author's intent — which is why every claim carries its source
  text, so a human or a second LLM judge can audit any flagged contradiction.

---

## 9. Module layout (suggested)

Keeping each file focused on one responsibility:

| File | Responsibility |
| --- | --- |
| `ir.py` | The typed IR: signature, terms, atoms, formulas, `Claim` |
| `typecheck.py` | `TypeChecker` — validate IR against a signature |
| `compile.py` | `Compiler` — lower IR to Z3 (one Z3 context per request) |
| `solve.py` | `solve_consistency`, `solve_entailment`, MUS minimization |
| `render.py` | Deterministic structural renderer (source of truth) |
| `explain.py` | Explanation builder (deterministic + optional LLM prose) |
| `prompts.py` | Extraction prompt fragments |
| `extract.py` | `Extractor` — signature & claim extraction (pydantic-ai) |
| `llm.py` | pydantic-ai agent factory + model config |
| `models.py` | Request/response models |
| `pipeline.py` | Orchestration (split manual, union, type-check, solve) |
| `api.py` | FastAPI app + wiring |

---

## 10. Worked example — stock news generation

A realistic guardrail run with three inputs:

1. **Generated stock news** (the `input_content` to be checked)
2. **Grounding data source** (the facts the article was supposed to stick to)
3. **Manual logic** — a compliance rule that **no personalized financial advice
   is given**

This example exercises both kinds of conflict the service detects: a **factual
contradiction** (a number the article got wrong vs. the source) and a **policy
violation** (advice the article gave that a manual rule forbids).

### 10.1 The three inputs

**(1) `input_content` — generated stock news**

> Acme Corp (ACME) reported second-quarter revenue of **$5.0 billion, up 12%
> year over year**. Given these strong results, **you should buy ACME shares
> for your retirement portfolio.**

**(2) `grounding_content` — data source**

> Acme Corp Q2 revenue was **$4.5 billion**, a **12%** increase year over year.
> Acme is listed on the NYSE under the ticker ACME.

**(3) `manual_logic` — compliance rule (pre-authored IR)**

> "The article must not give personalized financial advice."

### 10.2 Request

```json
{
  "input_content": "Acme Corp (ACME) reported second-quarter revenue of $5.0 billion, up 12% year over year. Given these strong results, you should buy ACME shares for your retirement portfolio.",
  "grounding_content": "Acme Corp Q2 revenue was $4.5 billion, a 12% increase year over year. Acme is listed on the NYSE under the ticker ACME.",
  "manual_logic": [
    {
      "id": "manual::no_personalized_advice",
      "source": "manual",
      "text": "The article must not give personalized financial advice.",
      "formula": {
        "node": "not",
        "arg": {
          "node": "concept",
          "concept": "gives_personalized_advice",
          "arg": { "node": "entity", "name": "the_article" }
        }
      }
    }
  ],
  "unique_names": true,
  "mode": "consistency",
  "explain_with_llm": true
}
```

### 10.3 Shared signature (extracted from all three inputs)

Note the signature includes `gives_personalized_advice` — a symbol that comes
from the **manual rule**, not the documents. Reading `manual_logic` during
signature extraction (§4, stage 1) is what guarantees the article's advice
sentence is normalized onto this same symbol so the two can collide.

```json
{
  "enums": [{ "name": "Exchange", "values": ["NYSE", "NASDAQ"] }],
  "entities": [
    { "name": "acme", "description": "Acme Corp, ticker ACME" },
    { "name": "the_article", "description": "the generated stock news article" }
  ],
  "concepts": [
    { "name": "gives_personalized_advice",
      "description": "the article tells the reader to take a personal investment action" }
  ],
  "roles": [],
  "attributes": [
    { "name": "revenue_q2_busd", "value_sort": { "kind": "Real" },
      "description": "Q2 revenue in billions USD" },
    { "name": "yoy_growth_pct", "value_sort": { "kind": "Real" },
      "description": "year-over-year revenue growth, percent" },
    { "name": "listed_on", "value_sort": { "kind": "Enum", "enum": "Exchange" },
      "description": "the exchange the company trades on" }
  ]
}
```

### 10.4 Extracted claims (union)

```json
[
  { "id": "input::1", "source": "input",
    "text": "second-quarter revenue of $5.0 billion",
    "formula": { "node": "compare", "op": "==",
      "left": { "node": "attr", "attribute": "revenue_q2_busd",
                "arg": { "node": "entity", "name": "acme" } },
      "right": { "node": "num", "value": 5.0 } } },

  { "id": "input::2", "source": "input",
    "text": "up 12% year over year",
    "formula": { "node": "compare", "op": "==",
      "left": { "node": "attr", "attribute": "yoy_growth_pct",
                "arg": { "node": "entity", "name": "acme" } },
      "right": { "node": "num", "value": 12.0 } } },

  { "id": "input::3", "source": "input",
    "text": "you should buy ACME shares for your retirement portfolio",
    "formula": { "node": "concept", "concept": "gives_personalized_advice",
      "arg": { "node": "entity", "name": "the_article" } } },

  { "id": "grounding::1", "source": "grounding",
    "text": "Q2 revenue was $4.5 billion",
    "formula": { "node": "compare", "op": "==",
      "left": { "node": "attr", "attribute": "revenue_q2_busd",
                "arg": { "node": "entity", "name": "acme" } },
      "right": { "node": "num", "value": 4.5 } } },

  { "id": "grounding::2", "source": "grounding",
    "text": "a 12% increase year over year",
    "formula": { "node": "compare", "op": "==",
      "left": { "node": "attr", "attribute": "yoy_growth_pct",
                "arg": { "node": "entity", "name": "acme" } },
      "right": { "node": "num", "value": 12.0 } } },

  { "id": "grounding::3", "source": "grounding",
    "text": "listed on the NYSE",
    "formula": { "node": "enum_eq", "attribute": "listed_on",
      "arg": { "node": "entity", "name": "acme" }, "value": "NYSE" } },

  { "id": "manual::no_personalized_advice", "source": "manual",
    "text": "The article must not give personalized financial advice.",
    "formula": { "node": "not", "arg": { "node": "concept",
      "concept": "gives_personalized_advice",
      "arg": { "node": "entity", "name": "the_article" } } } }
]
```

### 10.5 Response (`consistency` mode)

The conjunction is UNSAT because of **two independent conflicts**:

- **Factual:** `revenue_q2_busd(acme)` cannot equal both `5.0` (input) and `4.5`
  (grounding) — `revenue_q2_busd` is a function, so this is UNSAT for free.
- **Policy:** the article both `gives_personalized_advice(the_article)` (input)
  and is forbidden from doing so (manual rule).

The growth-rate and exchange claims are consistent and never enter a core.

```json
{
  "consistent": false,
  "status": "unsat",
  "contradicting_claim_ids": ["input::1", "grounding::1",
                              "input::3", "manual::no_personalized_advice"],
  "contradicting_claims": [
    { "id": "input::1", "source": "input", "text": "second-quarter revenue of $5.0 billion" },
    { "id": "grounding::1", "source": "grounding", "text": "Q2 revenue was $4.5 billion" },
    { "id": "input::3", "source": "input", "text": "you should buy ACME shares for your retirement portfolio" },
    { "id": "manual::no_personalized_advice", "source": "manual", "text": "The article must not give personalized financial advice." }
  ],
  "explanation": {
    "summary": "The generated article conflicts with the guardrails in two ways. (1) Factual: it reports Q2 revenue of $5.0B, but the source states $4.5B — a single company cannot have two different Q2 revenue figures. (2) Compliance: it tells the reader to buy ACME for their retirement, which is personalized financial advice that the manual rule explicitly forbids.",
    "rendered_claims": [
      "[input::1] (input) \"second-quarter revenue of $5.0 billion\" ⟦ revenue_q2_busd(acme) == 5.0 ⟧",
      "[grounding::1] (grounding) \"Q2 revenue was $4.5 billion\" ⟦ revenue_q2_busd(acme) == 4.5 ⟧",
      "[input::3] (input) \"you should buy ACME shares for your retirement portfolio\" ⟦ gives_personalized_advice(the_article) ⟧",
      "[manual::no_personalized_advice] (manual) \"...must not give personalized financial advice.\" ⟦ ¬ gives_personalized_advice(the_article) ⟧"
    ]
  },
  "signature": { "...": "see 10.3" },
  "claims": [ "...": "see 10.4" ]
}
```

> Implementation note: a single MUS shrink returns *one* locally-minimal core.
> Because this example has two independent contradictions, the solver loop
> should extract cores iteratively — report a core, then **drop one of its
> members and re-check** — to surface *both* the factual and the policy conflict
> rather than only the first one Z3 happens to return.

### 10.6 Same example in `entailment` mode

Here each input claim is judged against `grounding ∪ manual`:

| Claim | Verdict | Why |
| --- | --- | --- |
| `input::1` revenue `== 5.0` | **contradicted** | grounding fixes revenue at `4.5` |
| `input::2` growth `== 12.0` | **supported** | grounding entails it |
| `input::3` gives personalized advice | **contradicted** | manual rule forbids it |

If the article had instead claimed something the source neither stated nor
denied (e.g. *"Acme will raise its dividend next quarter"*), that claim would
come back **unsupported** with a counter-model witness — a world where the
grounding holds but the dividend claim is false — flagging it as an unfounded
(potentially hallucinated) statement.
