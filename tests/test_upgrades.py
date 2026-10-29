"""Tests for the three BadmintonTrackNet-inspired upgrades:

1. Temporal ensemble in ShuttleTracker (per-frame multi-window averaging).
2. Height-aware (floor) inpaint gate in TrajectoryRefiner.
3. Rule-based ShotClassifier (local, no cloud API).
"""

from __future__ import annotations

import numpy as np

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.analysis.shot_classifier import ShotEvent, classify_hit, classify_shots
from rallylens.vision.detect_track import Detection
from rallylens.vision.rectifier import TrajectoryRefiner
from rallylens.vision.shuttle_tracker import ShuttlePoint, _ensemble_weights

# ---------------------------------------------------------------------------
# 1. Temporal ensemble
# ---------------------------------------------------------------------------


def test_ensemble_weights_pyramid():
    w = _ensemble_weights(8)
    assert len(w) == 8
    expected = np.array([1, 2, 3, 4, 4, 3, 2, 1], dtype=float) / 20.0
    np.testing.assert_allclose(w, expected, atol=1e-9)
    assert abs(w.sum() - 1.0) < 1e-9
    # Centre frames must outweigh the edges.
    assert w[3] > w[0]


# ---------------------------------------------------------------------------
# 2. Floor gate in rectifier
# ---------------------------------------------------------------------------


def _make_refiner() -> TrajectoryRefiner:
    """Refiner with InpaintNet disabled (weights absent in test env) — the
    floor gate + linear fill must still work end to end."""
    return TrajectoryRefiner(weights_path=__import__("pathlib").Path("/nonexistent.pth"))


def test_floor_gate_drops_long_near_floor_gap():
    refiner = _make_refiner()
    # 60 frames, 720p; long missing run (frames 30..44) whose surrounding
    # observed frames sit near the floor (y > 0.95 * 720 = 684).
    pts = []
    for f in range(60):
        if 30 <= f < 45:
            continue  # the gap
        pts.append(ShuttlePoint(frame_idx=f, x=200 + f, y=700))  # near floor
    out = refiner.refine(pts, video_size=(1280, 720))
    emitted = {p.frame_idx for p in out}
    # The floor-gated gap must NOT be filled by linear interpolation.
    assert not any(30 <= f < 45 for f in emitted)
    # Outside frames survive.
    assert 0 in emitted and 59 in emitted


def test_floor_gate_leaves_mid_air_gap_untouched():
    refiner = _make_refiner()
    # Gap in the middle of the air (y small) is NOT floor-gated → linear fill
    # kicks in for a short gap.
    pts = []
    for f in range(40):
        if 18 <= f < 21:
            continue
        pts.append(ShuttlePoint(frame_idx=f, x=100 + f * 3, y=300))  # mid-air
    out = refiner.refine(pts, video_size=(1280, 720))
    emitted = {p.frame_idx for p in out}
    # Short gap gets linearly filled, not dropped.
    assert 19 in emitted


def test_floor_gate_short_gap_near_floor_is_filled():
    refiner = _make_refiner()
    # Short gap (3 frames) near the floor is a momentary occlusion — linear
    # fill should still answer it (gate only applies to long gaps).
    pts = []
    for f in range(40):
        if 18 <= f < 21:
            continue
        pts.append(ShuttlePoint(frame_idx=f, x=100 + f * 3, y=710))  # near floor
    out = refiner.refine(pts, video_size=(1280, 720))
    emitted = {p.frame_idx for p in out}
    assert 19 in emitted


# ---------------------------------------------------------------------------
# 3. Shot classification
# ---------------------------------------------------------------------------


def _track_around(frame_idx: int, speed: float, vy: float, n: int = 10) -> list[ShuttlePoint]:
    """Build a synthetic straight-ish trajectory around ``frame_idx``.

    speed: pixels/frame; vy: vertical delta per frame (positive = down).
    """
    pts: list[ShuttlePoint] = []
    for k in range(-n, n + 1):
        f = frame_idx + k
        if f == frame_idx:
            continue  # the hit frame itself often missing from the track
        x = 400 + k * speed
        y = 300 + k * vy
        pts.append(ShuttlePoint(frame_idx=f, x=int(x), y=int(y)))
    return pts


def _hit_event(frame_idx: int, x: int = 400, y: int = 300) -> ShuttleEvent:
    return ShuttleEvent(
        frame_idx=frame_idx,
        x=x,
        y=y,
        event_type="hit",
        rule="vy_reversal",
        confidence=0.9,
        all_rules=["vy_reversal"],
        features={},
    )


def _player(frame_idx: int, track_id: int = 1, cx: float = 400, cy: float = 300) -> Detection:
    return Detection(
        frame_idx=frame_idx,
        bbox_xyxy=(cx - 40, cy - 120, cx + 40, cy + 120),
        confidence=0.9,
        keypoints_xy=[(cx, cy)] * 17,
        keypoints_conf=[0.9] * 17,
        track_id=track_id,
    )


def test_smash_fast_falling():
    ev = _hit_event(100)
    track = _track_around(100, speed=40.0, vy=4.0)
    dets = [_player(96), _player(104)]
    shot = classify_hit(ev, track, dets, fps=30, video_size=(1280, 720))
    assert shot is not None
    assert shot.shot_type == "smash"
    assert shot.confidence > 0.8


def test_drive_fast_flat():
    ev = _hit_event(100)
    track = _track_around(100, speed=24.0, vy=0.5)
    dets = [_player(96), _player(104)]
    shot = classify_hit(ev, track, dets, fps=30, video_size=(1280, 720))
    assert shot is not None
    assert shot.shot_type == "drive"


def test_clear_rising():
    ev = _hit_event(100)
    # vy negative → shuttle rising after the hit.
    track = _track_around(100, speed=24.0, vy=-4.0)
    dets = [_player(96), _player(104)]
    shot = classify_hit(ev, track, dets, fps=30, video_size=(1280, 720))
    assert shot is not None
    assert shot.shot_type == "clear"


def test_drop_slow_falling():
    ev = _hit_event(100)
    track = _track_around(100, speed=8.0, vy=2.0)
    dets = [_player(96), _player(104)]
    shot = classify_hit(ev, track, dets, fps=30, video_size=(1280, 720))
    assert shot is not None
    assert shot.shot_type in ("drop", "net")


def test_classify_shots_filters_non_hit():
    landing = ShuttleEvent(
        frame_idx=200, x=100, y=500, event_type="landing", rule="speed_drop",
        confidence=0.8, all_rules=[], features={},
    )
    shots = classify_shots([landing], _track_around(200, 20, 2), [_player(200)])
    assert shots == []


def test_shot_event_model_roundtrip():
    shot = ShotEvent(
        frame_idx=10, player_track_id=2, shot_type="smash", confidence=0.85,
        speed_px=35.0, vy_after=3.0, x=100, y=200,
    )
    data = shot.model_dump()
    assert data["shot_type"] == "smash"
    assert ShotEvent.model_validate(data) == shot


def test_empty_inputs():
    assert classify_shots([], [], []) == []
    assert classify_hit(_hit_event(5), [], [], video_size=(1280, 720)) is None
