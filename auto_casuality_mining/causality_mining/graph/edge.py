"""Causal graph edge with refutation-grade metadata."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Edge:
    """Directed edge `source -> target` with the metadata required for promotion.

    The architecture doc requires that every promoted edge carry edge strength,
    confidence, supported lag, last validation date, and known confounders so
    that the policy service knows where it is safe to use.

    Attributes:
        source: source node id.
        target: target node id.
        lag: positive integer, in panel periods. `lag=k` means a 1-step
            backward change in the source at time t is associated with a
            k-step forward change in the target between t and t+k. Both sides
            of the relationship use the SAME `Change` encoder configured at
            curation time (e.g. PctChange features -> PctChange forward target).
        strength: signed magnitude of the estimated causal effect: per one unit
            of 1-step source change, how many units of k-step forward target
            change. Sign is meaningful for SCALAR/VECTOR sources.
        confidence: 0..1, derived from importance + refutation outcomes.
        importance: raw feature importance score (e.g. SHAP) before refutation.
        refutations: dict of refutation_name -> p-value-like score.
            Higher == more robust. Edges below the curator's threshold are
            dropped.
        confounders: ids of nodes treated as common-cause adjustments when the
            inference engine re-estimates the effect on a fresh DoWhy model.
        meta: free-form bag (e.g. last validation date, supported clusters).
    """

    source: str
    target: str
    lag: int
    strength: float
    confidence: float
    importance: float = 0.0
    refutations: dict[str, float] = field(default_factory=dict)
    confounders: tuple[str, ...] = ()
    meta: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple[str, str, int]:
        """A unique key for (source, target, lag) used for de-duplication."""
        return (self.source, self.target, self.lag)
