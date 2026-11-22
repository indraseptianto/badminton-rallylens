from __future__ import annotations

import numpy as np
import pytest

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.analysis.heatmaps import build_player_heatmaps, key_insights, zone_of
from rallylens.analysis.shot_classifier import ShotEvent
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.court_geometry import MARGIN, NET_Y
from rallylens.vision.detect_track import Detection
from rallylens.viz import render_heatmap_dashboard

FPS = 30.0
# pixel = court cm + (100, 100): homography is a pure shift, so expectations are exact
CORNERS = CourtCorners(
    top_left=(100, 100), top_right=(710, 100), bottom_left=(100, 1440), bottom_right=(710, 1440)
)


def det(frame: int, tid: int, cx: float, cy: float) -> Detection:
    px, py = 100 + cx, 100 + cy
    kps = [(0.0, 0.0)] * 17
    conf = [0.0] * 17
    for i in (15, 16):
        kps[i] = (px, py)
        conf[i] = 0.9
    return Detection(
        frame_idx=frame, bbox_xyxy=(px - 30, py - 150, px + 30, py), confidence=0.9,
        keypoints_xy=kps, keypoints_conf=conf, track_id=tid,
    )  # fmt: skip


def ev(frame: int, kind: str, x: float = 400, y: float = 500) -> ShuttleEvent:
    return ShuttleEvent(
        frame_idx=frame, x=x, y=y, event_type=kind, rule="t", confidence=1.0,
        all_rules=["t"], features={},
    )  # fmt: skip


def shot(frame: int, kind: str, tid: int = 1) -> ShotEvent:
    return ShotEvent(
        frame_idx=frame, player_track_id=tid, shot_type=kind, confidence=0.8,
        speed_px=10.0, vy_after=1.0, x=0, y=0,
    )  # fmt: skip


def scene() -> list[Detection]:
    out = []
    for f in range(300):  # player 1 walks 1 m/s toward the baseline for 10 s
        out.append(det(f, 1, 300, 900 + (100.0 / FPS) * f))
        out.append(det(f, 2, 300, 300))  # player 2 stands still
    return out


def test_movement_distance_and_speed() -> None:
    p1, p2 = build_player_heatmaps(scene(), [], [], CORNERS, FPS)
    assert p1.track_id == 1
    assert p1.distance_m == pytest.approx(10.0, abs=0.4)
    assert p1.avg_speed_ms == pytest.approx(1.0, abs=0.05)
    assert p2.distance_m == pytest.approx(0.0, abs=0.05)  # jitter-free stillness


def test_tracking_glitch_is_not_counted_as_distance() -> None:
    dets = [det(f, 1, 300, 900) for f in range(60)]
    dets[30] = det(30, 1, 300, 100)  # 8 m teleport for one frame
    (p,) = build_player_heatmaps(dets, [], [], CORNERS, FPS, top_n=1)
    assert p.distance_m is not None and p.distance_m < 3.0


def test_shots_rallies_and_endings() -> None:
    shots = [shot(30, "smash"), shot(60, "clear"), shot(90, "drop")]
    events = [ev(30, "hit"), ev(60, "hit"), ev(90, "hit"), ev(120, "landing", 400, 500)]
    p1, _ = build_player_heatmaps(scene(), shots, events, CORNERS, FPS)
    assert [s.shot_type for s in p1.shot_points] == ["smash", "clear", "drop"]
    assert all(s.rally_len == 3 for s in p1.shot_points)
    assert [s.shot_type for s in p1.last_shot_points] == ["drop"]
    assert p1.rally_lengths == [3] and len(p1.ending_points) == 1
    # shot position = the player's feet at that frame, not the aerial shuttle pixel
    assert p1.shot_points[0].x == MARGIN + 300


def test_zone_and_insights_honesty() -> None:
    assert zone_of(MARGIN + NET_Y) == "front" and zone_of(MARGIN + 1300) == "back"
    p1, _ = build_player_heatmaps(scene(), [], [], CORNERS, FPS)
    text = " ".join(key_insights(p1))
    assert "Most time spent" in text and "too few" in text.lower()


def test_dashboard_renders_with_and_without_shots() -> None:
    shots = [shot(30, "smash"), shot(60, "clear"), shot(90, "drop")]
    events = [ev(30, "hit"), ev(60, "hit"), ev(90, "hit"), ev(120, "landing")]
    full, _ = build_player_heatmaps(scene(), shots, events, CORNERS, FPS)
    bare, _ = build_player_heatmaps(scene(), [], [], CORNERS, FPS)
    a, b = render_heatmap_dashboard(full, "m"), render_heatmap_dashboard(bare, "m")
    assert a.shape == b.shape == (1024, 1536, 3) and a.dtype == np.uint8
    assert np.abs(a.astype(int) - b.astype(int)).sum() > 0  # panels differ with data


def test_invalid_fps_rejected() -> None:
    with pytest.raises(ValueError):
        build_player_heatmaps([], [], [], CORNERS, 0)


def test_pipeline_writes_one_png_per_player(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cv2

    from rallylens.pipeline import heatmap as hm

    monkeypatch.setattr(hm, "load_player_detections", lambda _v: scene())
    monkeypatch.setattr(hm, "load_court_corners", lambda _v: CORNERS)
    monkeypatch.setattr(hm, "load_events", lambda _v: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(hm, "load_shots", lambda _v: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(hm, "viz_heatmap_path", lambda v, t: tmp_path / f"{v}_{t}.png")
    paths = hm.run_heatmap_pipeline(tmp_path / "x.mp4", "vid", fps=FPS)
    assert sorted(p.name for p in paths) == ["vid_1.png", "vid_2.png"]
    assert cv2.imread(str(paths[0])).shape == (1024, 1536, 3)
