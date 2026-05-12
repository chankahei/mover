import json
from pathlib import Path

from doge.validator import verify_rulebase


def test_verifier_accepts_satisfiable_tree(tmp_path: Path) -> None:
    path = tmp_path / "ontology_rules.json"
    path.write_text(
        json.dumps(
            {
                "node_id": "root",
                "name": "Root",
                "is_leaf": False,
                "logic_gate": "OR",
                "children": [
                    {"node_id": "leaf_a", "name": "Leaf A", "is_leaf": True},
                    {"node_id": "leaf_b", "name": "Leaf B", "is_leaf": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = verify_rulebase(path)

    assert report.is_satisfiable
    assert report.leaf_count == 2
    assert report.hierarchy_constraint_count == 1


def test_verifier_reports_unsat_core_for_conflicting_constraints(tmp_path: Path) -> None:
    path = tmp_path / "ontology_rules.json"
    path.write_text(
        json.dumps(
            {
                "root": {
                    "node_id": "root",
                    "name": "Root",
                    "is_leaf": False,
                    "logic_gate": "OR",
                    "children": [
                        {
                            "node_id": "Mentions_Internal_Fund",
                            "name": "Mentions Internal Fund",
                            "is_leaf": True,
                        }
                    ],
                },
                "constraints": [
                    {
                        "rule_id": "Rule_042",
                        "kind": "implication",
                        "antecedents": ["Mentions_Internal_Fund"],
                        "consequent": "Approve_For_Publish",
                    },
                    {
                        "rule_id": "Rule_088",
                        "kind": "implication",
                        "antecedents": ["Mentions_Internal_Fund"],
                        "consequent": "Reject",
                    },
                    {
                        "rule_id": "Rule_099",
                        "kind": "mutex",
                        "terms": ["Approve_For_Publish", "Reject"],
                    },
                    {
                        "rule_id": "Rule_100",
                        "kind": "requires",
                        "terms": ["Mentions_Internal_Fund"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = verify_rulebase(path)

    assert report.status == "unsat"
    assert {"Rule_042", "Rule_088", "Rule_099", "Rule_100"}.issubset(set(report.unsat_core))
