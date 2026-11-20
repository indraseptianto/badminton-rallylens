from __future__ import annotations

import numpy as np
import pytest

from rallylens.analysis.events.detector import BounceDetector, ShuttleEvent
from rallylens.live.briefing import build_briefing, landing_zone
from rallylens.live.incremental import IncrementalEventDetector
from rallylens.live.queue import DropOldestQueue
from rallylens.live.segmenter import RallySegmenter
from rallylens.live.session import replay_live
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.shuttle_tracker import ShuttlePoint

FPS = 30.0
SIZE = (1280, 720)


def ev(frame: int, kind: str = "hit", x: float = 100, y: float = 100) -> ShuttleEvent:
    return ShuttleEvent(
        frame_idx=frame, x=x, y=y, event_type=kind, rule="t", confidence=1.0,
        all_rules=["t"], features={},
    )  # fmt: skip


def zigzag_track() -> list[ShuttlePoint]:
    """Shuttle bouncing between two ends: produces several direction reversals."""
    pts = []
    for f in range(0, 300):
        phase = (f // 25) % 2
        t = (f % 25) / 25
        x = 200 + (880 * (t if phase == 0 else 1 - t))
        y = 360 + 120 * np.sin(f / 9.0)
        pts.append(ShuttlePoint(frame_idx=f, x=int(x), y=int(y)))
    return pts


def test_queue_drops_oldest() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    assert [q.put(i) for i in (1, 2, 3)] == [False, False, True]
    assert q.dropped == 1 and q.get() == 2 and q.get() == 3 and q.get() is None
    with pytest.raises(ValueError):
        DropOldestQueue(0)


def test_incremental_equals_batch() -> None:
    track = zigzag_track()
    batch = BounceDetector(fps=FPS).detect_from_shuttle_points(
        np.array([p.frame_idx for p in track]),
        np.array([p.x for p in track]),
        np.array([p.y for p in track]),
        SIZE,
    )
    inc = IncrementalEventDetector(FPS, SIZE)
    got = []
    for p in track:
        inc.push(p)
        got += inc.poll(p.frame_idx)
    got += inc.flush()
    assert batch, "fixture must produce events"
    key = lambda e: (e.frame_idx, e.event_type)  # noqa: E731
    assert {key(e) for e in got} == {key(e) for e in batch}


def test_incremental_holds_back_unsettled_events() -> None:
    inc = IncrementalEventDetector(FPS, SIZE, settle_s=1.0)
    for p in zigzag_track()[:60]:
        inc.push(p)
    assert all(e.frame_idx <= 59 - 30 for e in inc.poll(59))


def test_segmenter_closes_on_landing_and_gap() -> None:
    seg = RallySegmenter(FPS, max_gap_s=2.0)
    assert seg.push(ev(10)) == [] and seg.push(ev(40)) == []
    (r,) = seg.push(ev(70, "landing"))
    assert r.shot_count == 2 and r.landing is not None and r.start_frame == 10
    seg.push(ev(200)), seg.push(ev(230))
    assert seg.advance(250) == []
    (r2,) = seg.advance(400)
    assert r2.landing is None and r2.end_frame == 230


def test_segmenter_drops_lone_hit() -> None:
    seg = RallySegmenter(FPS)
    seg.push(ev(10))
    assert seg.push(ev(60, "landing")) == []


def corners() -> CourtCorners:
    return CourtCorners(
        top_left=(300, 200), top_right=(980, 200), bottom_left=(100, 680), bottom_right=(1180, 680)
    )


def test_landing_zone_depth() -> None:
    c = corners()
    assert landing_zone(c, 640, 660) == "belakang"  # near the near baseline
    assert landing_zone(c, 640, 440) in ("depan", "tengah")
    assert landing_zone(c, 640, 20) is None  # far off the court


def test_briefing_honesty() -> None:
    empty = build_briefing([], [], fps=FPS, from_frame=0, to_frame=300)
    assert empty.rallies == 0 and empty.low_confidence
    seg = RallySegmenter(FPS)
    seg.push(ev(10)), seg.push(ev(40))
    rallies = seg.push(ev(70, "landing", 640, 660))
    b = build_briefing(rallies, [], fps=FPS, from_frame=0, to_frame=300)
    assert "n/a" in b.text() and b.landing_zones is None
    b2 = build_briefing(rallies, [], fps=FPS, from_frame=0, to_frame=300, corners=corners())
    assert b2.landing_zones is not None and b2.landing_zones["belakang"] == 1


def test_replay_emits_briefings_without_sleeping() -> None:
    sleeps: list[float] = []
    got = replay_live(
        zigzag_track(), fps=FPS, frame_size=SIZE, brief_every_s=4.0, speed=2.0,
        sleep=sleeps.append,
    )  # fmt: skip
    assert len(got) >= 2
    assert len(sleeps) == 300 and sleeps[0] == pytest.approx(1 / 60)
    assert replay_live([], fps=FPS, frame_size=SIZE) == []
