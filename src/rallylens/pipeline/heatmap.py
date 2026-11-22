"""Heatmap dashboards: wire saved artifacts → spatial statistics → one PNG per player."""

from __future__ import annotations

from pathlib import Path

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.analysis.heatmaps import build_player_heatmaps
from rallylens.analysis.shot_classifier import ShotEvent
from rallylens.common import ensure_dir, get_logger
from rallylens.pipeline.events import resolve_fps
from rallylens.pipeline.io import (
    load_court_corners,
    load_events,
    load_player_detections,
    load_shots,
    viz_heatmap_path,
)
from rallylens.viz import write_heatmap_dashboard

_log = get_logger(__name__)


def run_heatmap_pipeline(
    video_path: Path, video_id: str, fps: float | None = None, top_n: int = 2
) -> list[Path]:
    """Render a dashboard per tracked player; returns the written PNG paths.

    Needs player detections (with track IDs) and court corners. Events and shots are
    optional: panels that depend on them say so instead of being drawn empty.
    """
    detections = load_player_detections(video_id)
    corners = load_court_corners(video_id)
    events: list[ShuttleEvent] = []
    shots: list[ShotEvent] = []
    try:
        events = load_events(video_id)
    except FileNotFoundError:
        _log.warning("no events for %s — shot/rally panels will be empty (detect-events)", video_id)
    try:
        shots = load_shots(video_id)
    except FileNotFoundError:
        _log.warning("no shots for %s — shot panels will be empty (detect-shots)", video_id)

    players = build_player_heatmaps(
        detections, shots, events, corners, resolve_fps(video_path, fps), top_n=top_n
    )
    paths: list[Path] = []
    for p in players:
        out = viz_heatmap_path(video_id, p.track_id)
        ensure_dir(out.parent)
        write_heatmap_dashboard(p, video_id, out)
        paths.append(out)
    return paths
