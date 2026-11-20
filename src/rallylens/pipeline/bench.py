"""Throughput benchmark: can this machine keep up with a live feed?

Live coaching only works if the pipeline sustains more frames per second than
the camera produces. This module measures each stage on real frames of a video
and turns the numbers into a pass/fail verdict against a *real-time factor*
gate (sustained fps / video fps, default 1.2 for headroom).

The timing logic is pure (``summarize_stage`` / ``build_report``) so it is
unit-tested without models; ``run_benchmark`` wires it to the real stages.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from rallylens.common import get_logger, open_video, read_video_properties
from rallylens.vision.detect_track import coerce_tracker_name, make_frame_detector
from rallylens.vision.shuttle_tracker import SEQ_LEN, ShuttleTracker, describe_compute

_log = get_logger(__name__)

GATE_DEFAULT = 1.2

# Frames discarded before timing starts: first-call overhead (kernel compilation,
# allocator warm-up) and, for TrackNet, filling the SEQ_LEN-frame window.
DECODE_WARMUP = 2
POSE_WARMUP = 3
SHUTTLE_WARMUP = SEQ_LEN + 2

Verdict = Literal["pass", "pass_parallel", "fail"]
Step = Callable[[np.ndarray, int], object]


class StageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    frames: int  # timed frames (after warm-up)
    sustained_fps: float
    p50_ms: float
    p95_ms: float
    warmup_s: float


class BenchReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    device: str
    video_fps: float
    width: int
    height: int
    gate: float
    stages: list[StageResult]
    sequential_fps: float  # one worker runs every stage in turn
    parallel_fps: float  # one worker per stage: bounded by the slowest
    sequential_factor: float  # fps / video_fps
    parallel_factor: float
    verdict: Verdict
    suggested_stride: int  # frame stride a single worker would need (1 = none)


def summarize_stage(name: str, samples: Sequence[float], warmup: int) -> StageResult:
    """Per-frame seconds → throughput stats, discarding the first ``warmup`` samples.

    ``sustained_fps`` is timed frames / total time (what a live feed experiences),
    not 1 / median, so occasional slow frames are not hidden.
    """
    if len(samples) <= warmup:
        raise ValueError(f"{name}: need more than {warmup} samples, got {len(samples)}")
    timed = np.asarray(samples[warmup:], dtype=float)
    total = max(float(timed.sum()), 1e-9)
    return StageResult(
        name=name,
        frames=len(timed),
        sustained_fps=len(timed) / total,
        p50_ms=float(np.percentile(timed, 50)) * 1000.0,
        p95_ms=float(np.percentile(timed, 95)) * 1000.0,
        warmup_s=float(sum(samples[:warmup])),
    )


def build_report(
    stages: Sequence[StageResult],
    *,
    video_fps: float,
    size: tuple[int, int],
    device: str,
    gate: float = GATE_DEFAULT,
) -> BenchReport:
    """Combine stage results into sequential/parallel throughput and a verdict."""
    if not stages:
        raise ValueError("no stages to report on")
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")

    sequential_fps = 1.0 / sum(1.0 / s.sustained_fps for s in stages)
    parallel_fps = min(s.sustained_fps for s in stages)
    sequential_factor = sequential_fps / video_fps
    parallel_factor = parallel_fps / video_fps

    stride = 1
    verdict: Verdict
    if sequential_factor >= gate:
        verdict = "pass"
    elif parallel_factor >= gate:
        verdict = "pass_parallel"
    else:
        verdict = "fail"
        stride = math.ceil(gate / sequential_factor)

    return BenchReport(
        device=device,
        video_fps=video_fps,
        width=size[0],
        height=size[1],
        gate=gate,
        stages=list(stages),
        sequential_fps=sequential_fps,
        parallel_fps=parallel_fps,
        sequential_factor=sequential_factor,
        parallel_factor=parallel_factor,
        verdict=verdict,
        suggested_stride=stride,
    )


def run_benchmark(
    video_path: Path,
    *,
    frames: int = 60,
    gate: float = GATE_DEFAULT,
    pose: bool = True,
    shuttle: bool = True,
    shuttle_weights: Path | None = None,
    pose_weights: str = "yolo11s-pose.pt",
    pose_device: str | None = None,
    imgsz: int = 1280,
    tracker: str | None = "bytetrack",
    clock: Callable[[], float] = time.perf_counter,
) -> BenchReport:
    """Time decode + the selected stages on consecutive frames from the start of a video.

    Frames are processed one at a time through every stage (like a live loop),
    so memory stays flat and stages see realistic, interleaved load.

    Raises:
        RuntimeError: TrackNet weights are missing (a disabled tracker returns
            instantly and would report meaningless FPS).
        ValueError: nothing selected, or the video is too short.
    """
    if not (pose or shuttle):
        raise ValueError("nothing to benchmark: both pose and shuttle are skipped")

    props = read_video_properties(video_path)
    steps: dict[str, Step] = {}
    warmups: dict[str, int] = {"decode": DECODE_WARMUP}
    device = describe_compute()

    if shuttle:
        shuttle_tracker = ShuttleTracker(shuttle_weights)
        if not shuttle_tracker.enabled:
            raise RuntimeError(
                "TrackNet weights not found — refusing to benchmark a disabled tracker "
                "(it returns instantly and would report meaningless FPS). "
                "Pass --weights or place models/shuttle_tracknet.pth."
            )

        def shuttle_step(frame: np.ndarray, idx: int) -> object:
            return shuttle_tracker.detect(frame, idx)

        steps["shuttle"] = shuttle_step
        warmups["shuttle"] = SHUTTLE_WARMUP

    if pose:
        detector = make_frame_detector(
            pose_weights,
            imgsz=imgsz,
            tracker=coerce_tracker_name(tracker),
            device=pose_device,
        )

        def pose_step(frame: np.ndarray, idx: int) -> object:
            return detector(frame)

        steps["pose"] = pose_step
        warmups["pose"] = POSE_WARMUP
        device += f"; pose device: {pose_device or 'ultralytics default'}"

    total = frames + max(warmups.values())
    samples: dict[str, list[float]] = {name: [] for name in ("decode", *steps)}
    with open_video(video_path) as cap:
        for idx in range(total):
            t0 = clock()
            ok, frame = cap.read()
            samples["decode"].append(clock() - t0)
            if not ok:
                raise ValueError(
                    f"video has only {idx} frames; need {total} ({frames} timed + warm-up)"
                )
            for name, step in steps.items():
                t0 = clock()
                step(frame, idx)
                samples[name].append(clock() - t0)

    stages = [summarize_stage(name, samples[name], warmups[name]) for name in samples]
    return build_report(
        stages,
        video_fps=props.fps,
        size=(props.width, props.height),
        device=device,
        gate=gate,
    )


def format_report(report: BenchReport) -> str:
    """Human-readable summary of a ``BenchReport``."""
    lines = [
        "RallyLens throughput benchmark",
        f"video: {report.width}x{report.height} @ {report.video_fps:g} fps | "
        f"gate: {report.gate:g}x real-time",
        f"compute: {report.device}",
        "",
        f"{'stage':<8} {'frames':>6} {'fps':>8} {'p50 ms':>8} {'p95 ms':>8} {'warm-up s':>10}",
    ]
    for s in report.stages:
        lines.append(
            f"{s.name:<8} {s.frames:>6} {s.sustained_fps:>8.1f} {s.p50_ms:>8.1f} "
            f"{s.p95_ms:>8.1f} {s.warmup_s:>10.2f}"
        )
    lines += [
        "",
        f"one worker, stages in turn : {report.sequential_fps:>7.1f} fps "
        f"= {report.sequential_factor:.2f}x real-time",
        f"one worker per stage       : {report.parallel_fps:>7.1f} fps "
        f"= {report.parallel_factor:.2f}x real-time",
        "",
    ]
    if report.verdict == "pass":
        lines.append("verdict: PASS — a single worker keeps up with headroom.")
    elif report.verdict == "pass_parallel":
        lines.append(
            "verdict: PASS (parallel only) — needs one worker per stage; assumes the stages "
            "do not saturate one shared device."
        )
    else:
        lines.append(
            f"verdict: FAIL — below real-time. A frame stride of {report.suggested_stride} "
            "would fit on one worker (tracking accuracy at stride > 1 is unverified), or "
            "use a faster device."
        )
    return "\n".join(lines)
