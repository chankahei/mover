"""Challenge candidate edges with DoWhy refutation tests.

For each candidate `(treatment, outcome)`, build a single-edge `CausalModel`
in DoWhy v0.14, identify the effect, estimate it, then run several refuters:

- placebo_treatment_refuter
- random_common_cause
- data_subset_refuter

These are exactly the kind of checks the architecture doc requires before an
edge is promoted into production decisioning. Each candidate is converted into
an `Edge` whose `confidence` summarizes how well the candidate held up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from causality_mining.curation.candidates import CandidateEdge
from causality_mining.graph.edge import Edge


@dataclass(frozen=True)
class RefutationResult:
    """Outcome of running the DoWhy refuters on a single candidate."""

    estimate: float
    scores: dict[str, float]  # refuter -> robustness score in [0, 1]
    confounders: tuple[str, ...]


_DEFAULT_REFUTERS: tuple[str, ...] = (
    "placebo_treatment_refuter",
    "random_common_cause",
    "data_subset_refuter",
)


def _suppress_known_statsmodels_warnings() -> warnings.catch_warnings:
    """Silence noisy non-fatal statsmodels warnings in DoWhy refuters.

    DoWhy's internal statsmodels regressions can emit repeated runtime warnings
    (notably divide-by-zero in condition number diagnostics) on near-singular
    synthetic/bootstrap subsets. These warnings do not change our success/fail
    refutation handling and can overwhelm demo logs.
    """
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in scalar divide",
        category=RuntimeWarning,
        module=r"statsmodels\\.regression\\.linear_model",
    )
    return ctx


def _build_gml(treatment: str, outcome: str, confounders: Iterable[str]) -> str:
    """Tiny DAG with one treatment, one outcome, and optional common causes."""
    nodes = [treatment, outcome, *confounders]
    parts = ["digraph {"]
    for n in nodes:
        parts.append(f'  "{n}";')
    parts.append(f'  "{treatment}" -> "{outcome}";')
    for c in confounders:
        parts.append(f'  "{c}" -> "{treatment}";')
        parts.append(f'  "{c}" -> "{outcome}";')
    parts.append("}")
    return "\n".join(parts)


def _score_refutation(refuter_name: str, est_value: float, ref_value: float) -> float:
    """Map a refuter outcome to a robustness score in [0, 1].

    - `placebo_treatment_refuter`: pass if the effect collapses to ~0 under a
      random treatment, so high score == small ref_value.
    - other refuters: pass if the effect stays close to the original estimate.
    """
    if refuter_name == "placebo_treatment_refuter":
        denom = max(abs(est_value), 1e-9)
        return float(max(0.0, 1.0 - abs(ref_value) / denom))
    if abs(est_value) < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - abs(ref_value - est_value) / abs(est_value)))


def refute_candidate(
    candidate: CandidateEdge,
    panel: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    confounders: tuple[str, ...] = (),
    refuters: Iterable[str] = _DEFAULT_REFUTERS,
) -> RefutationResult | None:
    """Run DoWhy identify/estimate/refute on one candidate edge.

    Returns `None` if DoWhy fails on this candidate (which itself is informative:
    the candidate is dropped). Robustness scores are returned in `[0, 1]`.
    """
    from dowhy import CausalModel

    cols = [treatment_column, outcome_column, *confounders]
    df = panel[cols].dropna()
    if df.empty or df[treatment_column].nunique() < 2:
        return None

    try:
        warn_ctx = _suppress_known_statsmodels_warnings()
        model = CausalModel(
            data=df,
            treatment=treatment_column,
            outcome=outcome_column,
            graph=_build_gml(treatment_column, outcome_column, confounders),
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
        )
        warn_ctx.__exit__(None, None, None)
    except Exception:
        try:
            warn_ctx.__exit__(None, None, None)  # type: ignore[name-defined]
        except Exception:
            pass
        return None

    est_value = float(getattr(estimate, "value", 0.0))
    scores: dict[str, float] = {}
    for refuter_name in refuters:
        try:
            warn_ctx = _suppress_known_statsmodels_warnings()
            ref = model.refute_estimate(estimand, estimate, method_name=refuter_name)
            ref_value = float(getattr(ref, "new_effect", est_value))
            scores[refuter_name] = _score_refutation(refuter_name, est_value, ref_value)
            warn_ctx.__exit__(None, None, None)
        except Exception:
            try:
                warn_ctx.__exit__(None, None, None)  # type: ignore[name-defined]
            except Exception:
                pass
            scores[refuter_name] = 0.0

    return RefutationResult(estimate=est_value, scores=scores, confounders=tuple(confounders))


def to_edge(candidate: CandidateEdge, refutation: RefutationResult) -> Edge:
    """Build a graph `Edge` from a candidate + its refutation outcome."""
    confidence = float(np.clip(np.mean(list(refutation.scores.values()) or [0.0]), 0.0, 1.0))
    return Edge(
        source=candidate.source_series,
        target=candidate.target_series,
        lag=candidate.lag,
        strength=refutation.estimate,
        confidence=confidence,
        importance=candidate.importance,
        refutations=refutation.scores,
        confounders=refutation.confounders,
    )
