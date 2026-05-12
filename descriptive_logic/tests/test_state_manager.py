from pathlib import Path

from doge.models import LogicGate, RuleSuggestion
from doge.state_manager import OntologyState


def test_add_branch_saves_design_document_shape(tmp_path: Path) -> None:
    path = tmp_path / "ontology_rules.json"
    state = OntologyState.load(path)
    state.add_branch(
        parent_id="root",
        name="No financial advice",
        description="No financial advice",
        logic_gate=LogicGate.AND,
        suggestions=[
            RuleSuggestion(
                name="Mentions Monetary Target",
                condition="Text mentions a specific monetary goal.",
            ),
            RuleSuggestion(
                name="Suggests Asset Reallocation",
                condition="Text suggests changing asset allocation.",
            ),
        ],
    )

    state.save()
    loaded = OntologyState.load(path)

    root = loaded.rulebase.root
    assert root.children[0].name == "No financial advice"
    assert root.children[0].logic_gate == LogicGate.AND
    assert [child.is_leaf for child in root.children[0].children] == [True, True]
    assert root.children[0].children[0].node_id != root.children[0].children[1].node_id


def test_decompose_node_turns_existing_leaf_into_branch(tmp_path: Path) -> None:
    state = OntologyState.load(tmp_path / "ontology_rules.json")
    branch = state.add_branch(
        parent_id="root",
        name="No financial advice",
        description="No financial advice",
        logic_gate=LogicGate.AND,
        suggestions=[
            RuleSuggestion(
                name="Mentions Monetary Target",
                condition="Text mentions a specific monetary goal.",
            )
        ],
    )
    leaf = branch.children[0]

    state.decompose_node(
        node_id=leaf.node_id,
        logic_gate=LogicGate.OR,
        suggestions=[
            RuleSuggestion(
                name="Dollar Amount",
                condition="Text includes a dollar amount.",
            ),
            RuleSuggestion(
                name="Target Date",
                condition="Text includes a target date.",
            ),
        ],
    )

    decomposed = state.find(leaf.node_id)
    assert decomposed is not None
    assert decomposed.is_leaf is False
    assert decomposed.logic_gate == LogicGate.OR
    assert [child.name for child in decomposed.children] == ["Dollar Amount", "Target Date"]
