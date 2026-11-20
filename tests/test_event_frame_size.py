"""Event detection must use the real frame size, not one guessed from the track."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from rallylens.analysis.events.detector import BounceDetector
from rallylens.pipeline.events import resolve_video_size


def _confined_track() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast strokes confined to the left 40% / top half of a 1280x720 frame."""
    frames, xs, ys = [], [], []
    for i in range(100):
        seg, t = i // 20, (i % 20) / 20
        xs.append(1280 * (0.08 + 0.30 * (t if seg % 2 == 0 else 1 - t)))
        ys.append(330 - 150 * math.sin(math.pi * t))
        frames.append(i)
    return np.array(frames), np.array(xs), np.array(ys)


def _summary(events: list) -> list[tuple[int, str]]:
    return [(e.frame_idx, e.event_type) for e in events]


def test_true_frame_size_matches_the_model_space_oracle() -> None:
    f, x, y = _confined_track()
    det = BounceDetector(fps=30.0)
    true_size = det.detect_from_shuttle_points(f, x, y, frame_size=(1280, 720))
    # The same motion rendered in a 512x288 frame needs no rescale at all.
    oracle = det.detect_from_shuttle_points(f, x * 0.4, y * 0.4, frame_size=(512, 288))
    assert _summary(true_size) == _summary(oracle)
    assert sum(e.event_type == "hit" for e in true_size) == 4


def test_guessing_the_size_from_the_track_creates_false_hits() -> None:
    # Documents the old behaviour (still the fallback when the size is unknown):
    # a track that never reaches the frame edge looks bigger than it is.
    f, x, y = _confined_track()
    det = BounceDetector(fps=30.0)
    correct = det.detect_from_shuttle_points(f, x, y, frame_size=(1280, 720))
    guessed = det.detect_from_shuttle_points(f, x, y)
    hits = lambda ev: sum(e.event_type == "hit" for e in ev)  # noqa: E731
    assert hits(guessed) > hits(correct)


def test_event_coordinates_are_returned_in_original_pixels() -> None:
    f, x, y = _confined_track()
    events = BounceDetector(fps=30.0).detect_from_shuttle_points(f, x, y, frame_size=(1280, 720))
    assert all(0 <= e.x < 1280 and 0 <= e.y < 720 for e in events)
    assert all(e.x <= 1280 * 0.40 for e in events)  # still where the shuttle actually was


@pytest.mark.parametrize("size", [(0, 720), (1280, 0), (-1, -1)])
def test_invalid_frame_size_is_rejected(size: tuple[int, int]) -> None:
    f, x, y = _confined_track()
    with pytest.raises(ValueError, match="frame_size"):
        BounceDetector(fps=30.0).detect_from_shuttle_points(f, x, y, frame_size=size)


def test_resolve_video_size(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    assert resolve_video_size(video) == (64, 48)
    assert resolve_video_size(tmp_path / "missing.mp4") is None
