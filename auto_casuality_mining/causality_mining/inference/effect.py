"""Re-estimate one edge's causal effect on demand using DoWhy.

At inference time, the cached `Edge.strength` from the curated graph already
gives us a usable point estimate (cheap path). When the caller wants a fresh,
data-driven number that respects current confounders, this module rebuilds a
DoWhy `CausalModel` for that single edge and returns the new estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from causality_mining.curation.refute import _build_gml
from causality_mining.graph.edge import Edge


@dataclass(frozen=True)
class EdgeEffect:
    """Per-edge causal effect snapshot.

    Attributes:
        edge: the graph edge this effect belongs to.
        per_unit: estimated change in outcome per unit change of the treatment.
        sample_size: number of rows DoWhy actually used.
    """

    edge: Edge
    per_unit: float
    sample_size: int


def estimate_edge_effect(
    edge: Edge,
    panel: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounders: Iterable[str] | None = None,
) -> EdgeEffect:
    """Run DoWhy identify+estimate for a single edge against `panel`.

    Falls back to the cached `edge.strength` if DoWhy fails for any reason
    (so inference never blows up on a bad slice of history).
    """
    from dowhy import CausalModel

    confs = tuple(confounders) if confounders else edge.confounders
    cols = [treatment_column, outcome_column, *confs]
    df = panel[cols].dropna()
    if df.empty or df[treatment_column].nunique() < 2:
        return EdgeEffect(edge=edge, per_unit=edge.strength, sample_size=0)

    try:
        model = CausalModel(
            data=df,
            treatment=treatment_column,
            outcome=outcome_column,
            graph=_build_gml(treatment_column, outcome_column, confs),
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        per_unit = float(getattr(estimate, "value", edge.strength))
    except Exception:
        per_unit = edge.strength

    return EdgeEffect(edge=edge, per_unit=per_unit, sample_size=int(len(df)))
