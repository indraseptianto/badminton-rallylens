"""Throughput benchmark: pure timing maths + the run loop with fake stages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from rallylens.cli import cli
from rallylens.pipeline import bench
from rallylens.pipeline.bench import (
    StageResult,
    build_report,
    format_report,
    run_benchmark,
    summarize_stage,
)


def _stage(name: str, fps: float) -> StageResult:
    return StageResult(
        name=name, frames=50, sustained_fps=fps, p50_ms=1000 / fps, p95_ms=1500 / fps, warmup_s=0.0
    )


def _report(fps_by_stage: dict[str, float], video_fps: float = 30.0, gate: float = 1.2):
    return build_report(
        [_stage(n, f) for n, f in fps_by_stage.items()],
        video_fps=video_fps,
        size=(1280, 720),
        device="test",
        gate=gate,
    )


def _write_video(path: Path, fps: int = 30, frames: int = 40) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    return path


def _ticking_clock(step: float = 0.01) -> Callable[[], float]:
    t = {"now": 0.0}

    def clock() -> float:
        t["now"] += step
        return t["now"]

    return clock


# --- pure maths -----------------------------------------------------------------------------


def test_summarize_discards_warmup_and_uses_total_time() -> None:
    res = summarize_stage("x", [1.0, 1.0, 0.1, 0.1, 0.1, 0.1], warmup=2)
    assert res.frames == 4
    assert res.sustained_fps == pytest.approx(10.0)
    assert res.p50_ms == pytest.approx(100.0)
    assert res.warmup_s == pytest.approx(2.0)


def test_sustained_fps_is_not_hidden_by_median() -> None:
    # Mostly fast frames but one very slow one: throughput must reflect the slow frame.
    res = summarize_stage("x", [0.01] * 9 + [1.0], warmup=0)
    assert res.sustained_fps == pytest.approx(10 / 1.09)
    assert res.p50_ms == pytest.approx(10.0)


def test_summarize_requires_samples_beyond_warmup() -> None:
    with pytest.raises(ValueError, match="need more than"):
        summarize_stage("x", [0.1, 0.1], warmup=2)


def test_sequential_fps_combines_stages_in_turn() -> None:
    rep = _report({"pose": 60.0, "shuttle": 60.0}, video_fps=30.0)
    assert rep.sequential_fps == pytest.approx(30.0)
    assert rep.parallel_fps == pytest.approx(60.0)


def test_verdict_pass_single_worker() -> None:
    rep = _report({"pose": 200.0, "shuttle": 200.0})
    assert rep.verdict == "pass" and rep.suggested_stride == 1


def test_verdict_pass_parallel_only() -> None:
    # sequential 20 fps (0.67x) fails the 1.2 gate, but each stage alone is 40 fps (1.33x).
    rep = _report({"pose": 40.0, "shuttle": 40.0})
    assert rep.verdict == "pass_parallel" and rep.suggested_stride == 1


def test_verdict_fail_suggests_stride() -> None:
    # sequential 15 fps = 0.5x real-time → gate 1.2 needs ceil(1.2 / 0.5) = 3.
    rep = _report({"pose": 30.0, "shuttle": 30.0})
    assert rep.verdict == "fail" and rep.suggested_stride == 3


def test_build_report_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        build_report([], video_fps=30.0, size=(1, 1), device="t")
    with pytest.raises(ValueError):
        build_report([_stage("a", 10.0)], video_fps=0.0, size=(1, 1), device="t")


# --- run loop with fake stages ----------------------------------------------------------------


class _FakeTracker:
    enabled = True
    device_name = "cpu"

    def __init__(self, weights: Path | None = None) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[object]:
        self.calls += 1
        return []


class _DisabledTracker(_FakeTracker):
    enabled = False


def test_run_benchmark_times_each_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _write_video(tmp_path / "v.mp4", fps=30, frames=40)
    monkeypatch.setattr(bench, "ShuttleTracker", _FakeTracker)
    monkeypatch.setattr(bench, "make_frame_detector", lambda *a, **k: lambda frame: 1)
    rep = run_benchmark(video, frames=20, clock=_ticking_clock(0.01))
    by_name = {s.name: s for s in rep.stages}
    assert set(by_name) == {"decode", "shuttle", "pose"}
    # every timed interval spans exactly one clock tick of 10 ms → 100 fps
    assert all(s.sustained_fps == pytest.approx(100.0) for s in rep.stages)
    # 20 timed frames + the longest warm-up (shuttle) are read; each stage drops its own warm-up.
    total = 20 + bench.SHUTTLE_WARMUP
    assert by_name["shuttle"].frames == 20
    assert by_name["pose"].frames == total - bench.POSE_WARMUP
    assert by_name["decode"].frames == total - bench.DECODE_WARMUP
    assert rep.video_fps == pytest.approx(30.0) and (rep.width, rep.height) == (64, 48)


def test_run_benchmark_can_skip_pose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _write_video(tmp_path / "v.mp4")
    monkeypatch.setattr(bench, "ShuttleTracker", _FakeTracker)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("pose must not be loaded")

    monkeypatch.setattr(bench, "make_frame_detector", boom)
    rep = run_benchmark(video, frames=10, pose=False, clock=_ticking_clock())
    assert [s.name for s in rep.stages] == ["decode", "shuttle"]


def test_disabled_tracker_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _write_video(tmp_path / "v.mp4")
    monkeypatch.setattr(bench, "ShuttleTracker", _DisabledTracker)
    with pytest.raises(RuntimeError, match="weights not found"):
        run_benchmark(video, frames=10, pose=False)


def test_short_video_is_reported_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _write_video(tmp_path / "v.mp4", frames=15)
    monkeypatch.setattr(bench, "ShuttleTracker", _FakeTracker)
    with pytest.raises(ValueError, match="video has only"):
        run_benchmark(video, frames=60, pose=False)


def test_nothing_selected_is_an_error(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v.mp4")
    with pytest.raises(ValueError, match="nothing to benchmark"):
        run_benchmark(video, pose=False, shuttle=False)


# --- formatting + CLI -------------------------------------------------------------------------


def test_format_report_names_the_verdict() -> None:
    assert "PASS —" in format_report(_report({"pose": 200.0, "shuttle": 200.0}))
    assert "parallel only" in format_report(_report({"pose": 40.0, "shuttle": 40.0}))
    failed = format_report(_report({"pose": 30.0, "shuttle": 30.0}))
    assert "FAIL" in failed and "stride of 3" in failed


@pytest.mark.parametrize(("fps", "exit_code"), [(200.0, 0), (30.0, 1)])
def test_cli_exit_code_follows_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fps: float, exit_code: int
) -> None:
    video = _write_video(tmp_path / "v.mp4")
    monkeypatch.setattr("rallylens.cli.run_benchmark", lambda *a, **k: _report({"shuttle": fps}))
    out = tmp_path / "r.json"
    result = CliRunner().invoke(cli, ["bench", str(video), "--json-out", str(out)])
    assert result.exit_code == exit_code
    assert "verdict:" in result.output and out.exists()


def test_cli_reports_errors_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _write_video(tmp_path / "v.mp4")

    def fail(*a: object, **k: object) -> None:
        raise RuntimeError("TrackNet weights not found")

    monkeypatch.setattr("rallylens.cli.run_benchmark", fail)
    result = CliRunner().invoke(cli, ["bench", str(video)])
    assert result.exit_code == 1 and "weights not found" in result.output
