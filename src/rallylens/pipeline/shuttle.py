"""Shuttle tracking pipeline stage.

Opens a video, feeds every frame through ShuttleTracker, and persists
the detected ShuttlePoints to disk as a JSONL file.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from rallylens.common import ensure_dir, get_logger
from rallylens.config import TRACKS_DIR
from rallylens.pipeline.io import save_shuttle_track
from rallylens.vision.rectifier import TrajectoryRefiner
from rallylens.vision.shuttle_tracker import ShuttlePoint, ShuttleTracker

_log = get_logger(__name__)


def run_shuttle_pipeline(
    video_path: Path,
    video_id: str,
    weights_path: Path | None = None,
    rectify: bool = True,
) -> list[ShuttlePoint]:
    """Detect shuttlecock positions in every frame of a video.

    Args:
        video_path:   Path to the source video file.
        video_id:     Identifier used for the output subdirectory.
        weights_path: Optional explicit path to TrackNet weights.
                      Falls back to ``models/shuttle_tracknet.pth``.
        rectify:      When True (default), run trajectory rectification
                      (linear gap-fill + InpaintNet) on the raw detections.
                      Falls back to raw detections if the InpaintNet weights
                      are unavailable or the rectifier raises.

    Returns:
        list[ShuttlePoint] — one entry per estimated frame, sorted by frame_idx.
    """
    tracker = ShuttleTracker(weights_path)
    refiner = TrajectoryRefiner() if rectify else None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Dedup across overlapping sliding windows: each full window returns points
    # for SEQ_LEN frame indices; successive windows overlap by SEQ_LEN-1 frames.
    points_by_frame: dict[int, ShuttlePoint] = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for point in tracker.detect(frame, frame_idx):
            points_by_frame.setdefault(point.frame_idx, point)
        frame_idx += 1

    cap.release()

    all_points = [points_by_frame[k] for k in sorted(points_by_frame)]

    if refiner is not None:
        try:
            refined = refiner.refine(all_points, video_size)
            _log.info(
                "rectification: %d raw → %d refined points (%+d)",
                len(all_points),
                len(refined),
                len(refined) - len(all_points),
            )
            all_points = refined
        except Exception:  # noqa: BLE001 — rectification is best-effort
            _log.warning("rectification failed — keeping raw detections", exc_info=True)

    ensure_dir(TRACKS_DIR / video_id)
    out_path = save_shuttle_track(all_points, video_id)
    _log.info("saved %d shuttle points → %s", len(all_points), out_path)
    return all_points
