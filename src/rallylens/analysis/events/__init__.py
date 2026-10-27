"""Badminton rally event detection — ported from ZSHYC/BadmintonTrackNet.

Two-phase design from the original repo:

- **Phase 1 (rule-based, always on):** kinematics + rule bank
  (`KinematicsCalculator`, `BounceCandidateGenerator`, `BounceDetector`)
  detect ``landing`` / ``hit`` / ``out_of_frame`` events from the repaired
  shuttle trajectory. No learned model, no GPU — runs everywhere.
- **Phase 2 (BounceNet, optional):** a learned classifier that filters rule
  false-positives. NOT ported yet: it needs a trained checkpoint and, for the
  visual mode, frame patches — see ``docs/EVENTS.md`` before enabling.

Coordinate convention: the rule thresholds were tuned in TrackNet's model
space (512×288). This module therefore rescales input trajectories into
512×288 internally and maps event coordinates back to the original video
pixels, so raw 720p/1080p tracks work unchanged.
"""

from __future__ import annotations

from rallylens.analysis.events.detector import BounceDetector, ShuttleEvent
from rallylens.analysis.events.kinematics import KinematicsCalculator
from rallylens.analysis.events.rules import (
    BounceCandidateGenerator,
    RallyBounceDetector,
)

# Unified event type constants (single source of truth).
EVENT_TYPES = ["landing", "hit", "out_of_frame", "other"]
"""Event types:
- landing:      shuttle hits the court floor
- hit:          racket strikes the shuttle
- out_of_frame: shuttle leaves the visible frame
- other:        anything else
"""

# Primary rules decide the event type (mutually exclusive per frame).
PRIMARY_RULES: dict[str, float] = {
    "vy_reversal": 0.85,
    "vx_reversal": 0.80,
    "acceleration_peak": 0.75,
    "visibility_drop": 0.85,
    "speed_drop": 0.75,
    "visibility_drop_edge": 0.50,
    "trajectory_end": 0.70,
}

# Auxiliary rules boost confidence of an existing candidate.
AUXILIARY_RULES: dict[str, float] = {
    "y_local_max": 0.05,
    "speed_local_max": 0.05,
}

__all__ = [
    "KinematicsCalculator",
    "BounceCandidateGenerator",
    "RallyBounceDetector",
    "BounceDetector",
    "ShuttleEvent",
    "EVENT_TYPES",
    "PRIMARY_RULES",
    "AUXILIARY_RULES",
]
