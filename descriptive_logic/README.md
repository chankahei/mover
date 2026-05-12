# DOGE: Deterministic Ontology Guardrail Engine

DOGE turns vague compliance restrictions into a deterministic boolean ontology and verifies the resulting rulebase with Z3.

## Install

```bash
cd descriptive_logic
uv sync --extra dev
```

DOGE loads `.env` from `descriptive_logic` or any parent folder. Set `OPENAI_API_KEY`,
`DOGE_LLM_MODEL`, and `OPENAI_BASE_URL` there for OpenAI-compatible endpoints such as Poe.
`DOGE_LLM_BASE_URL`, `POE_BASE_URL`, and `BASE_URL` are also accepted aliases for the base URL.

## Build An Ontology

```bash
uv run doge.py build -f ontology_rules.json
```

Use `--offline` to exercise the terminal flow without LLM credentials:

```bash
uv run doge.py build --offline -f ontology_rules.json
```

Useful builder commands:

- At `/Root`, type a new top-level restriction to create a new branch.
- At any child node, press Enter to decompose that node itself.
- `/ls` lists children with numbers and node IDs.
- `/cd 1` moves into the first child; `/cd node_id` moves to a specific node.
- `/tree` shows the current ontology.
- `/up` moves to the parent node.
- `/leaf` marks the current node as terminal.
- `/save` writes the current state.
- `/exit` saves and exits.

## Verify An Ontology

```bash
uv run doge.py verify -f ontology_rules.json
```

The verifier accepts the root-node schema from the design document. It also accepts an extended rulebase form with additional compliance constraints:

```json
{
  "root": {
    "node_id": "rule_001",
    "name": "Financial Advice Block",
    "description": "Any content constituting personalized financial advice.",
    "is_leaf": false,
    "logic_gate": "OR",
    "children": []
  },
  "constraints": [
    {
      "rule_id": "Rule_042",
      "kind": "implication",
      "antecedents": ["Mentions_Internal_Fund"],
      "consequent": "Approve_For_Publish"
    },
    {
      "rule_id": "Rule_088",
      "kind": "mutex",
      "terms": ["Approve_For_Publish", "Reject"]
    }
  ]
}
```

Supported compliance rule kinds are `implication`, `mutex`, `requires`, and `forbids`.
