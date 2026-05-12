from __future__ import annotations

from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from doge.models import BreakdownSuggestion, LogicGate, RuleSuggestion
from doge.ontologist import OfflineOntologist, Ontologist
from doge.state_manager import OntologyState
from doge.validator import verify_rulebase


app = typer.Typer(help="Deterministic Ontology Guardrail Engine (DOGE).")
console = Console()


@app.command()
def build(
    file: Path = typer.Option(Path("ontology_rules.json"), "--file", "-f", help="Ontology file."),
    offline: bool = typer.Option(False, "--offline", help="Use deterministic local suggestions."),
    model: str | None = typer.Option(None, "--model", help="Pydantic AI model name."),
) -> None:
    """Run the interactive ontology builder."""

    state = OntologyState.load(file)
    ontologist = OfflineOntologist() if offline else Ontologist(model=model)
    session = PromptSession()
    current_id = "root"

    console.print("[bold green][DOGE][/bold green] Deterministic Ontology Guardrail Engine")
    console.print("=" * 46)
    console.print("Commands: /exit, /up, /tree, /ls, /cd <number|node_id>, /leaf, /save")

    while True:
        current = state.find(current_id) or state.rulebase.root
        _print_path(state, current.node_id)
        rule_text = _prompt_for_rule(session, current)

        if not rule_text and current.node_id == "root":
            continue
        if rule_text == "/exit":
            state.save()
            console.print(f"Saving to {file}... [green]Success.[/green]")
            break
        if rule_text == "/save":
            state.save()
            console.print(f"Saving to {file}... [green]Success.[/green]")
            continue
        if rule_text == "/tree":
            _print_tree(state.rulebase.root)
            continue
        if rule_text == "/ls":
            _print_children(current)
            continue
        if rule_text == "/up":
            parent = state.parent_of(current.node_id)
            current_id = parent.node_id if parent else "root"
            continue
        if rule_text.startswith("/cd"):
            target_id = _resolve_cd_target(state, current, rule_text)
            if target_id is None:
                console.print("[yellow]Unknown child or node id.[/yellow]")
            else:
                current_id = target_id
            continue
        if rule_text == "/leaf":
            if current.node_id == "root":
                console.print("[yellow]Root cannot be marked as a leaf.[/yellow]")
                continue
            state.mark_leaf(current.node_id)
            state.save()
            console.print(f"Marked {current.name!r} as terminal.")
            parent = state.parent_of(current.node_id)
            current_id = parent.node_id if parent else "root"
            continue

        target_text = rule_text or _node_target_text(current)
        console.print(f"\nQuerying LLM for Description Logic breakdown of {target_text!r}...")
        suggestion = ontologist.suggest(target_text)
        kept = _curate_suggestions(session, suggestion)
        if not kept:
            console.print("[yellow]No predicates selected; nothing saved.[/yellow]")
            continue

        if current.node_id == "root":
            branch = state.add_branch(
                parent_id=current.node_id,
                name=target_text,
                description=target_text,
                logic_gate=suggestion.logic_gate,
                suggestions=kept,
            )
        else:
            branch = state.decompose_node(
                node_id=current.node_id,
                logic_gate=suggestion.logic_gate,
                suggestions=kept,
            )
        state.save()
        console.print(f"\nSaving to {file}... [green]Success.[/green]")

        next_id = _choose_next_node(session, branch)
        if next_id == "DONE":
            current_id = state.parent_of(branch.node_id).node_id if state.parent_of(branch.node_id) else "root"
        elif next_id == "UP":
            parent = state.parent_of(branch.node_id)
            current_id = parent.node_id if parent else "root"
        else:
            current_id = next_id


@app.command()
def verify(
    file: Path = typer.Option(Path("ontology_rules.json"), "--file", "-f", help="Ontology file."),
) -> None:
    """Verify an ontology rulebase with Z3."""

    console.print("[bold green][DOGE][/bold green] Initializing Z3 SMT Solver...")
    report = verify_rulebase(file)
    console.print(f"Loading {report.leaf_count} boolean leaf nodes...")
    console.print(f"Loading {report.hierarchy_constraint_count} hierarchical constraints...")
    console.print(f"Loading {report.compliance_constraint_count} compliance constraints...")
    console.print("Running satisfiability check...\n")

    if report.is_satisfiable:
        console.print("[green]No contradictions detected.[/green]")
        console.print("Result: SATISFIABLE")
        raise typer.Exit(0)

    console.print("[bold red][!] CONFLICT DETECTED [!][/bold red]")
    console.print("Result: UNSATISFIABLE\n")
    console.print("Contradiction found in Core Logic:")
    for item in report.unsat_core:
        console.print(f"- {item}")
    raise typer.Exit(1)


def _prompt_for_rule(session: PromptSession, current) -> str:
    if current.node_id == "root":
        return session.prompt("Enter a restriction to decompose: ").strip()
    return session.prompt("Press Enter to decompose this node, or type a command: ").strip()


def _node_target_text(node) -> str:
    if node.description:
        return f"{node.name}: {node.description}"
    return node.name


def _curate_suggestions(session: PromptSession, suggestion: BreakdownSuggestion) -> list[RuleSuggestion]:
    console.print(
        f'To determine the target rule, the following sub-conditions are proposed '
        f"([bold]{suggestion.logic_gate.value}[/bold] logic):"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Predicate")
    table.add_column("Condition")
    for index, condition in enumerate(suggestion.conditions, start=1):
        table.add_row(str(index), condition.name, condition.condition)
    console.print(table)

    answer = session.prompt("Select conditions to keep (e.g. 1,2), or type 'custom': ").strip()
    if answer.lower() == "custom":
        return _custom_suggestions(session)
    keep: list[RuleSuggestion] = []
    selected = {part.strip() for part in answer.split(",") if part.strip()}
    for index, condition in enumerate(suggestion.conditions, start=1):
        if str(index) in selected:
            keep.append(condition)
    return keep


def _custom_suggestions(session: PromptSession) -> list[RuleSuggestion]:
    console.print("Enter custom predicates. Leave name blank when done.")
    suggestions: list[RuleSuggestion] = []
    while True:
        name = session.prompt("Predicate name: ").strip()
        if not name:
            break
        condition = session.prompt("Binary condition: ").strip() or name
        suggestions.append(RuleSuggestion(name=name, condition=condition))
    return suggestions


def _choose_next_node(session: PromptSession, branch) -> str:
    console.print(f"\nPath: /{branch.name}")
    console.print("Selected:")
    for index, child in enumerate(branch.children, start=1):
        console.print(f"{index}. {child.name}")
    answer = session.prompt("Recurse into number, [U]p, or [D]one: ").strip().upper()
    if answer == "U":
        return "UP"
    if answer == "D":
        return "DONE"
    if answer.isdigit():
        index = int(answer) - 1
        if 0 <= index < len(branch.children):
            return branch.children[index].node_id
    return "DONE"


def _print_children(node) -> None:
    if not node.children:
        console.print("[yellow]No children at this node.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Node ID")
    table.add_column("Name")
    for index, child in enumerate(node.children, start=1):
        table.add_row(str(index), child.node_id, child.name)
    console.print(table)


def _resolve_cd_target(state: OntologyState, current, command: str) -> str | None:
    parts = command.split(maxsplit=1)
    if len(parts) == 1:
        _print_children(current)
        return None
    target = parts[1].strip()
    if target.isdigit():
        index = int(target) - 1
        if 0 <= index < len(current.children):
            return current.children[index].node_id
        return None
    if state.find(target):
        return target
    for child in current.children:
        if child.name == target or child.node_id == target:
            return child.node_id
    return None


def _print_path(state: OntologyState, node_id: str) -> None:
    path = state.path_to(node_id)
    display = "/" + "/".join(node.name.replace(" ", "_") for node in path)
    console.print(f"\n[bold]Current Path:[/bold] {display}")


def _print_tree(root) -> None:
    tree = Tree(root.name)
    _add_tree_nodes(tree, root)
    console.print(tree)


def _add_tree_nodes(tree: Tree, node) -> None:
    for child in node.children:
        label = f"{child.name} [{child.node_id}]"
        if child.logic_gate:
            label += f" ({child.logic_gate.value})"
        branch = tree.add(label)
        _add_tree_nodes(branch, child)
