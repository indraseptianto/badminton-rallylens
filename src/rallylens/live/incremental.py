"""Incremental event detection on a growing shuttle track.

Rules look at frames on both sides of a candidate, so an event is only trustworthy
once ``settle_frames`` newer frames exist. ``poll`` re-runs the batch detector on the
buffer and releases events that have settled; ``flush`` releases the rest. Running the
same detector keeps live and batch results identical by construction.
"""

from __future__ import annotations

import numpy as np

from rallylens.analysis.events.detector import BounceDetector, ShuttleEvent
from rallylens.vision.shuttle_tracker import ShuttlePoint

MERGE_TOL = 3  # frames; same as BounceDetector's default merge_window


class IncrementalEventDetector:
    def __init__(
        self,
        fps: float,
        frame_size: tuple[int, int] | None = None,
        settle_s: float = 0.3,
        window_s: float = 60.0,
    ) -> None:
        self._detector = BounceDetector(fps=fps)
        self._frame_size = frame_size
        self._settle = max(1, round(settle_s * fps))
        self._window = max(self._settle * 4, round(window_s * fps))
        self._points: list[ShuttlePoint] = []
        self._emitted: list[ShuttleEvent] = []

    @property
    def points(self) -> list[ShuttlePoint]:
        return self._points

    def push(self, point: ShuttlePoint) -> None:
        self._points.append(point)

    def _is_new(self, ev: ShuttleEvent) -> bool:
        tol = MERGE_TOL
        return not any(
            e.event_type == ev.event_type and abs(e.frame_idx - ev.frame_idx) <= tol
            for e in self._emitted[-20:]
        )

    def _run(self, upto: int | None) -> list[ShuttleEvent]:
        if not self._points:
            return []
        idxs = [p.frame_idx for p in self._points]
        found = self._detector.detect_from_shuttle_points(
            np.array(idxs),
            np.array([p.x for p in self._points]),
            np.array([p.y for p in self._points]),
            self._frame_size,
        )
        fresh = [e for e in found if (upto is None or e.frame_idx <= upto) and self._is_new(e)]
        self._emitted.extend(fresh)
        return fresh

    def poll(self, current_frame: int) -> list[ShuttleEvent]:
        """Events whose context is complete as of ``current_frame``."""
        fresh = self._run(current_frame - self._settle)
        cutoff = current_frame - self._window
        if self._points and self._points[0].frame_idx < cutoff:
            self._points = [p for p in self._points if p.frame_idx >= cutoff]
        return fresh

    def flush(self) -> list[ShuttleEvent]:
        return self._run(None)
