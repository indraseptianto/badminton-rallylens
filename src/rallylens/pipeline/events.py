"""Rally event detection pipeline stage.

Reads a shuttle track (from ``detect-shuttle``), runs the rule-based event
detector (ported from ZSHYC/BadmintonTrackNet, Phase 1 without BounceNet),
and persists the detected landing / hit / out_of_frame events as JSONL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rallylens.analysis.events.detector import BounceDetector, ShuttleEvent
from rallylens.common import ensure_dir, get_logger, read_video_properties
from rallylens.config import EVENTS_DIR
from rallylens.pipeline.io import save_events
from rallylens.vision.shuttle_tracker import ShuttlePoint

_log = get_logger(__name__)

DEFAULT_FPS = 30.0


def resolve_fps(video_path: Path, fps: float | None = None) -> float:
    """Return ``fps`` if given, else the video's real frame rate.

    Event kinematics scale velocities by ``fps / 30``, so assuming 30 fps on a
    60 fps clip silently changes which events are detected. Falls back to
    ``DEFAULT_FPS`` only when the container reports nothing usable.
    """
    if fps is not None:
        return float(fps)
    try:
        detected = float(read_video_properties(video_path).fps)
    except (FileNotFoundError, RuntimeError):
        _log.warning("could not read fps from %s — assuming %.0f", video_path, DEFAULT_FPS)
        return DEFAULT_FPS
    return detected if detected > 0 else DEFAULT_FPS


def resolve_video_size(video_path: Path) -> tuple[int, int] | None:
    """Real ``(width, height)`` of the video, or None when it cannot be read."""
    try:
        props = read_video_properties(video_path)
    except (FileNotFoundError, RuntimeError):
        _log.warning("could not read frame size from %s — inferring from the track", video_path)
        return None
    if props.width <= 0 or props.height <= 0:
        return None
    return (props.width, props.height)


def run_events_pipeline(
    video_path: Path,
    video_id: str,
    track: list[ShuttlePoint] | None = None,
    fps: float | None = None,
) -> list[ShuttleEvent]:
    """Detect rally events (landing / hit / out_of_frame) from a shuttle track.

    Args:
        video_path: Path to the source video (used for fps detection).
        video_id:   Identifier used for the output subdirectory.
        track:      Shuttle track; when None, loads from disk.
        fps:        Video framerate for the kinematics calculator; when None it is
                    read from ``video_path`` (see ``resolve_fps``).

    Returns:
        list[ShuttleEvent] — detected events, sorted by frame_idx.
    """
    if track is None:
        from rallylens.pipeline.io import load_shuttle_track

        track = load_shuttle_track(video_id)
        _log.info("loaded %d shuttle points from disk", len(track))

    if not track:
        _log.warning("empty shuttle track — no events; save empty file")
        ensure_dir(EVENTS_DIR / video_id)
        out_path = save_events([], video_id)
        return []

    frame_idxs = np.array([p.frame_idx for p in track])
    xs = np.array([p.x for p in track], dtype=np.float64)
    ys = np.array([p.y for p in track], dtype=np.float64)

    detector = BounceDetector(fps=resolve_fps(video_path, fps))
    events = detector.detect_from_shuttle_points(
        frame_idxs, xs, ys, frame_size=resolve_video_size(video_path)
    )
    events.sort(key=lambda e: e.frame_idx)

    ensure_dir(EVENTS_DIR / video_id)
    out_path = save_events(events, video_id)
    counts: dict[str, int] = {}
    for e in events:
        counts[e.event_type] = counts.get(e.event_type, 0) + 1
    _log.info(
        "saved %d events (%s) → %s",
        len(events),
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none",
        out_path,
    )
    return events
