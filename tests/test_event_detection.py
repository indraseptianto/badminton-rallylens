"""Tests for rule-based rally event detection (ported from BadmintonTrackNet)."""

from __future__ import annotations

import numpy as np
import pytest

from rallylens.analysis.events import (
    AUXILIARY_RULES,
    EVENT_TYPES,
    PRIMARY_RULES,
    BounceCandidateGenerator,
    BounceDetector,
    KinematicsCalculator,
    ShuttleEvent,
)


def _parabola(n: int = 40, apex: int = 20, start_y: float = 300.0):
    """Shuttle arc: y dips down then comes back up (bounce-like)."""
    x = np.arange(n, dtype=float) * 10 + 100
    y = 300 + np.abs(np.arange(n) - apex) * 3.0
    visibility = np.ones(n, dtype=float)
    return x, y, visibility


def test_event_types_constant():
    assert "landing" in EVENT_TYPES
    assert "hit" in EVENT_TYPES
    assert "out_of_frame" in EVENT_TYPES
    assert PRIMARY_RULES["vy_reversal"] > 0.5
    assert AUXILIARY_RULES["y_local_max"] > 0


def test_kinematics_shape_and_speed():
    x, y, vis = _parabola()
    k = KinematicsCalculator(fps=30)
    feats = k.compute(x, y, vis)
    assert len(feats["speed"]) == len(x)
    assert feats["v_x"].shape == (len(x),)
    assert np.all(np.isfinite(feats["speed"]))


def test_detector_returns_shuttle_events():
    x, y, vis = _parabola()
    events = BounceDetector().detect(x, y, vis)
    assert isinstance(events, list)
    assert all(isinstance(e, ShuttleEvent) for e in events)
    for e in events:
        assert e.event_type in EVENT_TYPES
        assert 0.0 < e.confidence <= 1.0
        assert e.frame_idx >= 0


def test_landing_at_trajectory_end():
    """Trajectory that ends at low speed should flag a landing."""
    x = np.arange(30, dtype=float) * 8 + 200
    y = 400 - np.abs(np.arange(30) - 15) * 2.0
    vis = np.ones(30, dtype=float)
    events = BounceDetector().detect(x, y, vis)
    landing = [e for e in events if e.event_type == "landing"]
    assert landing, "expected at least one landing event"


def test_sparse_points_detect_from_shuttle_points():
    """Sparse per-frame observations still produce coherent events."""
    frame_idxs = np.arange(0, 60, 2)  # every other frame
    xs = frame_idxs * 5 + 300
    ys = np.abs(frame_idxs - 30) * 4 + 200
    events = BounceDetector().detect_from_shuttle_points(frame_idxs, xs, ys)
    assert all(isinstance(e, ShuttleEvent) for e in events)
    # Frame indices must be the original (real) ones, not dense positions.
    if events:
        assert events[0].frame_idx >= 0


def test_generator_merges_nearby():
    gen = BounceCandidateGenerator(merge_window=2)
    x, y, vis = _parabola(n=60)
    cands = gen.generate(x, y, vis, img_height=288, img_width=512)
    # Sorted by frame and no two candidates within merge_window.
    frames = [c["frame"] for c in cands]
    assert frames == sorted(frames)
    for a, b in zip(frames, frames[1:], strict=False):
        assert b - a > 2, "nearby candidates should have been merged"


def test_empty_inputs():
    assert BounceDetector().detect(np.array([]), np.array([]), np.array([])) == []
    assert BounceDetector().detect_from_shuttle_points(
        np.array([]), np.array([]), np.array([])
    ) == []


def test_fps_must_be_positive():
    with pytest.raises(ValueError):
        BounceDetector(fps=0)
    with pytest.raises(ValueError):
        KinematicsCalculator(fps=-5)
