"""Shot classification pipeline stage.

Reads the shuttle track + rally events + player detections and persists the
classified stroke types (smash / drive / drop / clear / net) as JSONL.
"""

from __future__ import annotations

from pathlib import Path

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.analysis.shot_classifier import ShotEvent, classify_shots
from rallylens.common import ensure_dir, get_logger
from rallylens.config import SHOTS_DIR
from rallylens.pipeline.io import save_shots
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint

_log = get_logger(__name__)


def run_shots_pipeline(
    video_path: Path,
    video_id: str,
    track: list[ShuttlePoint] | None = None,
    events: list[ShuttleEvent] | None = None,
    detections: list[Detection] | None = None,
    fps: int = 30,
) -> list[ShotEvent]:
    """Classify stroke types from events + track + detections.

    Args:
        video_path: Path to the source video (used for video size & fps).
        video_id:   Identifier for the output subdirectory.
        track:      Shuttle track (list[ShuttlePoint]); loads from disk when None.
        events:     Rally events (list[ShuttleEvent]); loads from disk when None.
        detections: Player detections (list[Detection]); loads from disk when None.
        fps:        Video framerate.

    Returns:
        list[ShotEvent] — a classified stroke per hit event, sorted by frame.
    """
    if track is None:
        from rallylens.pipeline.io import load_shuttle_track

        track = load_shuttle_track(video_id)
    if events is None:
        from rallylens.pipeline.io import load_events

        events = load_events(video_id)
    if detections is None:
        from rallylens.pipeline.io import load_player_detections

        detections = load_player_detections(video_id)

    if not events:
        _log.info("no events — saving empty shots file")
        ensure_dir(SHOTS_DIR / video_id)
        out = save_shots([], video_id)
        return []

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    video_size = None
    if cap.isOpened():
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    shots = classify_shots(events, track, detections, fps=fps, video_size=video_size)
    ensure_dir(SHOTS_DIR / video_id)
    out = save_shots(shots, video_id)
    counts: dict[str, int] = {}
    for s in shots:
        counts[s.shot_type] = counts.get(s.shot_type, 0) + 1
    _log.info(
        "saved %d shots (%s) → %s",
        len(shots),
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none",
        out,
    )
    return shots
