"""fps must come from the video, not a hard-coded 30."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from rallylens.analysis.events.detector import BounceDetector
from rallylens.pipeline.events import DEFAULT_FPS, resolve_fps


def _write_video(path: Path, fps: int, frames: int = 4) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    return path


def test_resolve_fps_reads_video(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v60.mp4", fps=60)
    assert resolve_fps(video) == pytest.approx(60.0)


def test_resolve_fps_explicit_value_wins(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v60.mp4", fps=60)
    assert resolve_fps(video, 24.0) == 24.0


def test_resolve_fps_missing_file_falls_back(tmp_path: Path) -> None:
    assert resolve_fps(tmp_path / "nope.mp4") == DEFAULT_FPS


def test_fractional_fps_is_accepted() -> None:
    # 29.97 used to be truncated to 29 by the orchestrator.
    assert BounceDetector(fps=29.97).fps == pytest.approx(29.97)
