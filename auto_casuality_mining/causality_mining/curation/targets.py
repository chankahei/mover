"""Specification of which timeseries are prediction targets.

The architecture doc lists CTR, order ticket open, executed trade, etc. as the
targets the causal graph should ultimately explain. Curation needs to know
which ids these are because:
  - It only fits predictive models for *target* nodes.
  - Targets are not allowed to be parents of non-target source nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetSpec:
    """Marks one timeseries id as a prediction target.

    Attributes:
        series_id: id of the target timeseries (e.g. "ctr", "trade_executed").
        kind: ML task to use when fitting the curation predictor.
            "regression" for SCALAR targets (e.g. CTR rate, dwell time).
            "classification" for binary / categorical targets (e.g. trade_yes_no).
        max_lag: largest lag (in panel periods) considered when proposing
            candidate parents for this target.
        min_lag: smallest lag (typically 1, to avoid same-timestamp leakage).
    """

    series_id: str
    kind: str = "regression"
    max_lag: int = 6
    min_lag: int = 1
    extras: dict[str, str] = field(default_factory=dict)
