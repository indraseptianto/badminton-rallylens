from __future__ import annotations

import numpy as np
import pytest

from rallylens.analysis.events.detector import BounceDetector, ShuttleEvent
from rallylens.analysis.rallies import RallySegmenter
from rallylens.live.briefing import build_briefing, landing_zone
from rallylens.live.incremental import IncrementalEventDetector
from rallylens.live.queue import DropOldestQueue
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


def test_runner_feeds_engine_and_counts_drops() -> None:
    from rallylens.live.runner import run_live
    from rallylens.live.session import LiveEngine

    track = {p.frame_idx: p for p in zigzag_track()}
    blank = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = ((i, blank) for i in range(300))

    def detect(_f: np.ndarray, idx: int) -> list[ShuttlePoint]:
        return [track[idx]]

    engine = LiveEngine(fps=FPS, frame_size=SIZE, brief_every_s=4.0)
    dropped = run_live(frames, detect, engine, queue_size=400, flush=lambda: [])
    assert dropped == 0 and len(engine.finish()) >= 2


def test_runner_drops_when_consumer_is_slow() -> None:
    import time

    from rallylens.live.runner import run_live
    from rallylens.live.session import LiveEngine

    blank = np.zeros((2, 2, 3), dtype=np.uint8)
    engine = LiveEngine(fps=FPS, frame_size=SIZE)

    def slow(_f: np.ndarray, _i: int) -> list[ShuttlePoint]:
        time.sleep(0.01)
        return []

    dropped = run_live(((i, blank) for i in range(200)), slow, engine, queue_size=2)
    assert dropped > 0


def test_frame_source_reads_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import cv2

    from rallylens.vision.frame_source import FrameSource

    path = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    with FrameSource(str(path)) as src:
        got = [i for i, _ in src]
        assert got == [0, 1, 2, 3, 4]
        assert (src.info.width, src.info.height) == (64, 48) and src.info.fps == 25.0
    with pytest.raises(RuntimeError):
        FrameSource(str(tmp_path / "missing.mp4"))


def test_engine_attributes_shots_to_players() -> None:
    from rallylens.live.session import LiveEngine
    from rallylens.vision.detect_track import Detection

    track = zigzag_track()
    engine = LiveEngine(fps=FPS, frame_size=SIZE, brief_every_s=100.0)
    for p in track:
        # a player standing wherever the shuttle is → every hit is attributable
        det = Detection(
            frame_idx=p.frame_idx, bbox_xyxy=(p.x - 20, p.y - 20, p.x + 20, p.y + 20),
            confidence=0.9, keypoints_xy=[], keypoints_conf=[], track_id=7,
        )  # fmt: skip
        engine.feed(p.frame_idx, [p], [det])
    (b,) = engine.finish()
    assert set(b.player_shots) <= {7} and "n/a (pose" not in b.text()


def test_briefing_without_pose_says_na() -> None:
    seg = RallySegmenter(FPS)
    seg.push(ev(10)), seg.push(ev(40))
    b = build_briefing(seg.push(ev(70, "landing")), [], fps=FPS, from_frame=0, to_frame=100)
    assert b.player_shots == {} and "pose belum aktif" in b.text()


def test_frame_source_reconnects_live_but_not_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import cv2

    from rallylens.vision.frame_source import FrameSource

    path = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(3):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    slept: list[float] = []
    with FrameSource(str(path), sleep=slept.append) as src:
        assert len(list(src)) == 3 and slept == []  # file end: no reconnect
    src2 = FrameSource(str(path), reconnect_attempts=2, sleep=slept.append)
    src2._live = True  # simulate a stream whose source keeps reopening fine
    it = iter(src2)
    assert [i for i, _ in zip(range(4), it, strict=False)] == [0, 1, 2, 3]
    assert src2.reconnects == 1 and slept == [1.0]
    src2.__exit__()
