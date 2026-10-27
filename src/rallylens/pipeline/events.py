"""Rally event detection pipeline stage.

Reads a shuttle track (from ``detect-shuttle``), runs the rule-based event
detector (ported from ZSHYC/BadmintonTrackNet, Phase 1 without BounceNet),
and persists the detected landing / hit / out_of_frame events as JSONL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rallylens.analysis.events.detector import BounceDetector, ShuttleEvent
from rallylens.common import ensure_dir, get_logger
from rallylens.config import EVENTS_DIR
from rallylens.pipeline.io import save_events
from rallylens.vision.shuttle_tracker import ShuttlePoint

_log = get_logger(__name__)


def run_events_pipeline(
    video_path: Path,
    video_id: str,
    track: list[ShuttlePoint] | None = None,
    fps: int = 30,
) -> list[ShuttleEvent]:
    """Detect rally events (landing / hit / out_of_frame) from a shuttle track.

    Args:
        video_path: Path to the source video (used for fps detection).
        video_id:   Identifier used for the output subdirectory.
        track:      Shuttle track; when None, loads from disk.
        fps:        Video framerate used by the kinematics calculator.

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

    detector = BounceDetector(fps=fps)
    events = detector.detect_from_shuttle_points(frame_idxs, xs, ys)
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
