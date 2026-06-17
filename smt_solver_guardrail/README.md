# SMT Solver Guardrail

A FastAPI service that checks whether LLM-generated content is logically
consistent with its grounding source, by compiling natural-language claims into
a strongly-typed description-logic IR and discharging them with the **Z3** SMT
solver. When a contradiction exists it returns a **minimal** set of conflicting
claims plus a **natural-language explanation by default**.

See [`DESIGN.md`](./DESIGN.md) for the full design and [`TYPES.md`](./TYPES.md)
for the authoritative type listing.

## How it works

1. **Shared signature** — an LLM reads the input document, the grounding
   document, *and* the manual rules together and emits one canonical signature
   (entities, concepts, roles, attributes, enums).
2. **Claim extraction** — claims are extracted from each document, constrained
   to that signature, then unioned with the manual rules. Manual rules may be
   pre-authored `Claim`s **or** natural-language strings (the latter converted
   to claims by the LLM). All LLM calls use **pydantic-ai** typed agents.
3. **Type check** — every claim is validated against the signature before
   anything reaches Z3 (malformed IR → HTTP 422).
4. **Solve** — `consistency` (default) checks satisfiability and reports a
   minimized UNSAT core; `entailment` labels each input claim
   `supported` / `unsupported` / `contradicted`.
5. **Explain** — the minimal core is rendered deterministically and (by
   default) paraphrased by an LLM confined to the already-proven core.

## Setup

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
```

LLM access is read from the environment (a local `.env` is auto-loaded):

- `OPENAI_API_KEY` — API key
- `BASE_URL` — optional OpenAI-compatible base URL
- `GUARDRAIL_LLM_MODEL` — model id (falls back to `DOGE_LLM_MODEL`, then `gpt-4o-mini`)

## Run the server

```bash
uv run uvicorn guardrail.api:app --reload
```

Then `POST /guardrail`:

```json
{
  "input_content": "Acme reported Q2 revenue of $5.0B, up 12% YoY. You should buy ACME for your retirement.",
  "grounding_content": "Acme Q2 revenue was $4.5B, a 12% increase YoY. Listed on the NYSE.",
  "manual_logic": [
    "The article must not give personalized financial advice."
  ],
  "unique_names": true,
  "mode": "consistency",
  "explain_with_llm": true
}
```

## Offline demo (no LLM)

The stock-news example from the design doc, with hand-built claims:

```bash
uv run python -m examples.stock_news
```

## Tests

```bash
uv run pytest -q
```

## Module layout

| File | Responsibility |
| --- | --- |
| `guardrail/ir.py` | The typed IR: signature, terms, atoms, formulas, `Claim` |
| `guardrail/typecheck.py` | `TypeChecker` — validate IR against a signature |
| `guardrail/compile.py` | `Compiler` — lower IR to Z3 (one Z3 context per request) |
| `guardrail/solve.py` | consistency, entailment, MUS minimization |
| `guardrail/render.py` | deterministic structural renderer |
| `guardrail/explain.py` | explanation builder (deterministic + optional LLM) |
| `guardrail/prompts.py` | extraction prompt fragments |
| `guardrail/extract.py` | `Extractor` — signature & claim extraction (pydantic-ai) |
| `guardrail/llm.py` | pydantic-ai agent factory + model config |
| `guardrail/models.py` | request/response models |
| `guardrail/pipeline.py` | orchestration |
| `guardrail/api.py` | FastAPI app |
