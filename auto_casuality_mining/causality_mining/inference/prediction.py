"""Result objects returned by the inference engine."""
from __future__ import annotations

from dataclasses import dataclass, field

from causality_mining.inference.effect import EdgeEffect


@dataclass(frozen=True)
class TargetPrediction:
    """One target's predicted response to an event.

    Attributes:
        target_id: target series id.
        delta: predicted change in the target relative to its current level
            attributable to the event (sum over edges along the source -> target
            paths). Sign is meaningful for SCALAR/VECTOR targets.
        baseline: target's current level (last observed value).
        predicted: baseline + delta. Convenience for downstream consumers.
        confidence: 0..1, mean of the contributing edges' confidences.
        contributing_edges: per-edge effects that built up `delta`.
    """

    target_id: str
    delta: float
    baseline: float
    predicted: float
    confidence: float
    contributing_edges: tuple[EdgeEffect, ...] = ()


@dataclass(frozen=True)
class Prediction:
    """Bundle of `TargetPrediction`s, one per target reachable from the event."""

    event_series_id: str
    targets: dict[str, TargetPrediction] = field(default_factory=dict)

    def best(self) -> TargetPrediction | None:
        """Highest-|delta| target, useful for ranking article opportunities."""
        if not self.targets:
            return None
        return max(self.targets.values(), key=lambda t: abs(t.delta))
