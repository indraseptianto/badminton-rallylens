"""Replay-as-live: drive the live components from a saved shuttle track."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.analysis.rallies import Rally, RallySegmenter
from rallylens.analysis.shot_classifier import ShotEvent, classify_hit
from rallylens.live.briefing import Briefing, build_briefing
from rallylens.live.incremental import IncrementalEventDetector
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint


class LiveEngine:
    """Consumes shuttle points frame by frame and emits a briefing every interval."""

    def __init__(
        self,
        *,
        fps: float,
        frame_size: tuple[int, int],
        brief_every_s: float = 60.0,
        corners: CourtCorners | None = None,
        on_briefing: Callable[[Briefing], None] | None = None,
    ) -> None:
        self.fps = fps
        self.frame_size = frame_size
        self.corners = corners
        self.on_briefing = on_briefing
        self.briefings: list[Briefing] = []
        self._interval = max(1, round(brief_every_s * fps))
        self._detector = IncrementalEventDetector(fps, frame_size)
        self._segmenter = RallySegmenter(fps)
        self._rallies: list[Rally] = []
        self._shots: list[ShotEvent] = []
        self._dets: list[Detection] = []
        self._start: int | None = None
        self._last = 0

    def _handle(self, events: list[ShuttleEvent]) -> None:
        for ev in events:
            if ev.event_type == "hit":
                shot = classify_hit(
                    ev, self._detector.points, self._dets, self.fps, self.frame_size
                )
                if shot is not None:
                    self._shots.append(shot)
            self._rallies.extend(self._segmenter.push(ev))

    def _brief(self, to_frame: int) -> None:
        assert self._start is not None
        b = build_briefing(
            self._rallies, self._shots, fps=self.fps, from_frame=self._start,
            to_frame=to_frame, corners=self.corners,
        )  # fmt: skip
        self.briefings.append(b)
        if self.on_briefing:
            self.on_briefing(b)
        self._rallies.clear()
        self._shots.clear()
        self._start = to_frame

    def feed(
        self,
        frame_idx: int,
        points: Sequence[ShuttlePoint],
        detections: Sequence[Detection] = (),
    ) -> None:
        if self._start is None:
            self._start = frame_idx
        self._last = frame_idx
        for p in points:
            self._detector.push(p)
        self._dets.extend(detections)
        keep = frame_idx - 600  # shots are attributed ±15 frames; bound memory
        if self._dets and self._dets[0].frame_idx < keep:
            self._dets = [d for d in self._dets if d.frame_idx >= keep]
        self._handle(self._detector.poll(frame_idx))
        self._rallies.extend(self._segmenter.advance(frame_idx))
        if frame_idx - self._start >= self._interval:
            self._brief(frame_idx)

    def finish(self) -> list[Briefing]:
        if self._start is None:
            return self.briefings
        self._handle(self._detector.flush())
        self._rallies.extend(self._segmenter.advance(self._last + 10_000))
        if self._last - self._start > 0:
            self._brief(self._last)
        return self.briefings


def replay_live(
    track: Sequence[ShuttlePoint],
    *,
    fps: float,
    frame_size: tuple[int, int],
    brief_every_s: float = 60.0,
    corners: CourtCorners | None = None,
    speed: float = 0.0,
    on_briefing: Callable[[Briefing], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Briefing]:
    """Feed ``track`` frame by frame; ``speed`` 1.0 = real time, 0 = as fast as possible."""
    if not track:
        return []
    by_frame: dict[int, list[ShuttlePoint]] = {}
    for p in track:
        by_frame.setdefault(p.frame_idx, []).append(p)
    engine = LiveEngine(
        fps=fps, frame_size=frame_size, brief_every_s=brief_every_s, corners=corners,
        on_briefing=on_briefing,
    )  # fmt: skip
    for f in range(min(by_frame), max(by_frame) + 1):
        if speed > 0:
            sleep(1.0 / (fps * speed))
        engine.feed(f, by_frame.get(f, []))
    return engine.finish()
