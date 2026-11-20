"""Shot classifier: rules at the reference setup, invariance elsewhere."""

from __future__ import annotations

import pytest

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.analysis.shot_classifier import classify_hit, classify_shots
from rallylens.vision.shuttle_tracker import ShuttlePoint

HIT = 10


def _event(x: int, y: int, event_type: str = "hit", frame: int = HIT) -> ShuttleEvent:
    return ShuttleEvent(
        frame_idx=frame,
        x=x,
        y=y,
        event_type=event_type,
        rule="test",
        confidence=0.9,
        all_rules=[],
        features={},
    )


def _track(
    hit_xy: tuple[float, float],
    v_before: tuple[float, float],
    v_after: tuple[float, float],
    *,
    step: int = 1,
    n: int = 4,
) -> list[ShuttlePoint]:
    """Shuttle at constant velocity (px per *frame*) into and out of the hit.

    ``step`` is frames between samples, so a 60 fps clip of the same physical
    shot has half the per-frame velocity and twice as many samples.
    """
    hx, hy = hit_xy
    pts = []
    for k in range(n, 0, -1):
        pts.append(
            ShuttlePoint(
                frame_idx=HIT - k * step,
                x=int(hx - v_before[0] * k * step),
                y=int(hy - v_before[1] * k * step),
            )
        )
    for k in range(1, n + 1):
        pts.append(
            ShuttlePoint(
                frame_idx=HIT + k * step,
                x=int(hx + v_after[0] * k * step),
                y=int(hy + v_after[1] * k * step),
            )
        )
    return pts


def test_smash_at_reference_setup() -> None:
    # 720p, 30 fps: fast, downward, mid-frame contact.
    track = _track((640, 400), (10, -5), (30, 15))
    shot = classify_hit(_event(640, 400), track, [], fps=30.0, video_size=(1280, 720))
    assert shot is not None and shot.shot_type == "smash"


def test_net_shot_is_slow_and_high_in_frame() -> None:
    track = _track((640, 150), (4, 1), (4, 1))
    shot = classify_hit(_event(640, 150), track, [], fps=30.0, video_size=(1280, 720))
    assert shot is not None and shot.shot_type == "net"


def test_insufficient_context_returns_none() -> None:
    track = [ShuttlePoint(frame_idx=9, x=0, y=0), ShuttlePoint(frame_idx=11, x=5, y=5)]
    assert classify_hit(_event(3, 3), track, [], fps=30.0, video_size=(1280, 720)) is None


def test_only_hit_events_are_classified() -> None:
    track = _track((640, 400), (10, -5), (30, 15))
    events = [_event(640, 400, "landing", frame=HIT), _event(640, 400, "hit", frame=HIT)]
    shots = classify_shots(events, track, [], fps=30.0, video_size=(1280, 720))
    assert [s.frame_idx for s in shots] == [HIT]


@pytest.mark.parametrize("width,height", [(1920, 1080), (640, 360)])
def test_same_physical_shot_same_label_across_resolutions(width: int, height: int) -> None:
    f = width / 1280
    track = _track((640 * f, 400 * f), (10 * f, -5 * f), (30 * f, 15 * f))
    shot = classify_hit(
        _event(int(640 * f), int(400 * f)), track, [], fps=30.0, video_size=(width, height)
    )
    assert shot is not None and shot.shot_type == "smash"


def test_same_physical_shot_same_label_at_60fps() -> None:
    # Half the displacement per frame, twice the frames: identical real-world shot.
    track = _track((640, 400), (5, -2.5), (15, 7.5), step=1, n=8)
    shot = classify_hit(_event(640, 400), track, [], fps=60.0, video_size=(1280, 720))
    assert shot is not None and shot.shot_type == "smash"


@pytest.mark.parametrize("width,height", [(1920, 1080), (640, 360)])
def test_drop_is_not_mislabelled_at_other_resolutions(width: int, height: int) -> None:
    # A slow falling drop (15.8 px/f at 720p) used to become "unknown" at 1080p.
    f = width / 1280
    track = _track((640 * f, 400 * f), (12 * f, 2 * f), (15 * f, 5 * f))
    shot = classify_hit(
        _event(int(640 * f), int(400 * f)), track, [], fps=30.0, video_size=(width, height)
    )
    assert shot is not None and shot.shot_type == "drop"


def test_persisted_speed_stays_in_raw_pixels() -> None:
    track = _track((960, 600), (15, -7), (45, 22))  # 1080p
    shot = classify_hit(_event(960, 600), track, [], fps=30.0, video_size=(1920, 1080))
    assert shot is not None and shot.speed_px == pytest.approx(50.09, abs=0.1)
