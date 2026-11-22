from __future__ import annotations

import pytest

from rallylens.analysis.rallies import Rally
from rallylens.live.rally_summary import build_rally_summary
from rallylens.live.session import replay_live

from .test_heatmaps import CORNERS, FPS, det, ev, shot
from .test_live import SIZE, zigzag_track


def make_rally() -> Rally:
    hits = [ev(30, "hit"), ev(60, "hit"), ev(90, "hit")]
    return Rally(start_frame=30, end_frame=120, hits=hits, landing=ev(120, "landing", 640, 660))


def scene_dets() -> list:  # type: ignore[type-arg]
    out = []
    for f in range(0, 150):
        out.append(det(f, 1, 300, 900 + (100.0 / FPS) * f))  # 1 m/s
        out.append(det(f, 2, 300, 300))  # standing
    return out


def test_summary_counts_distance_and_history() -> None:
    shots = [shot(30, "clear", 1), shot(60, "smash", 2), shot(90, "drop", 1), shot(500, "net", 1)]
    s = build_rally_summary(make_rally(), 1, shots, scene_dets(), fps=FPS, corners=CORNERS)
    assert [h.number for h in s.history] == [1, 2, 3]  # the frame-500 shot is not in this rally
    assert s.players[1].shots == 2 and s.players[2].shots == 1
    assert s.players[1].distance_m == pytest.approx(3.0, abs=0.5)
    assert s.players[2].distance_m == pytest.approx(0.0, abs=0.05)
    assert s.shot_types == {"clear": 1, "smash": 1, "drop": 1}
    assert s.ended_by_landing and s.landing_zone in ("depan", "tengah", "belakang")
    text = s.text()
    assert "Reli #1" in text and "1.P1 clear → 2.P2 smash → 3.P1 drop" in text


def test_summary_is_honest_without_pose_or_calibration() -> None:
    shots = [shot(30, "clear", None), shot(60, "smash", None)]  # type: ignore[arg-type]
    s = build_rally_summary(make_rally(), 2, shots, [], fps=FPS, corners=None)
    assert s.players == {} and s.landing_zone is None
    text = s.text()
    assert "pose belum aktif" in text and "Zona jatuh: n/a" in text
    no_cal = build_rally_summary(make_rally(), 3, [shot(30, "clear", 1)], scene_dets(), fps=FPS)
    assert no_cal.players[1].distance_m is None and "n/a" in no_cal.text()


def test_unfinished_rally_is_flagged() -> None:
    r = Rally(start_frame=0, end_frame=60, hits=[ev(0, "hit"), ev(30, "hit")], landing=None)
    s = build_rally_summary(r, 1, [], [], fps=FPS)
    assert not s.ended_by_landing and "shuttle hilang" in s.text()
    assert "n/a (klasifikasi" in s.text()


def test_engine_emits_rallies_and_shots_independent_of_brief_interval() -> None:
    def run(brief_s: float) -> tuple[list, list]:  # type: ignore[type-arg]
        rallies: list = []  # type: ignore[type-arg]
        shots: list = []  # type: ignore[type-arg]
        replay_live(
            zigzag_track(), fps=FPS, frame_size=SIZE, brief_every_s=brief_s,
            on_rally=rallies.append, on_shot=shots.append,
        )  # fmt: skip
        return rallies, shots

    a_r, a_s = run(1.0)  # briefing resets state many times mid-rally
    b_r, b_s = run(1000.0)
    assert a_r and a_s and [s.frame_idx for s in a_s] == [s.frame_idx for s in b_s]
    assert [r.index for r in a_r] == list(range(1, len(a_r) + 1))
    assert [len(r.history) for r in a_r] == [len(r.history) for r in b_r]
