"""Replay-as-live: drive the live components from a saved shuttle track."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from rallylens.analysis.shot_classifier import ShotEvent, classify_hit
from rallylens.live.briefing import Briefing, build_briefing
from rallylens.live.incremental import IncrementalEventDetector
from rallylens.live.segmenter import Rally, RallySegmenter
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.shuttle_tracker import ShuttlePoint


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
    """Feed ``track`` frame by frame; emit a briefing every ``brief_every_s`` of video time.

    ``speed`` = playback multiple (1.0 = real time, 0 = as fast as possible).
    """
    if not track:
        return []
    by_frame: dict[int, list[ShuttlePoint]] = {}
    for p in track:
        by_frame.setdefault(p.frame_idx, []).append(p)
    first, last = min(by_frame), max(by_frame)

    detector = IncrementalEventDetector(fps, frame_size)
    segmenter = RallySegmenter(fps)
    window_rallies: list[Rally] = []
    window_shots: list[ShotEvent] = []
    briefings: list[Briefing] = []
    interval = max(1, round(brief_every_s * fps))
    window_start = first

    def handle(events: list) -> None:  # type: ignore[type-arg]
        for ev in events:
            if ev.event_type == "hit":
                shot = classify_hit(ev, detector.points, [], fps, frame_size)
                if shot is not None:
                    window_shots.append(shot)
            window_rallies.extend(segmenter.push(ev))

    def brief(to_frame: int) -> None:
        nonlocal window_start
        b = build_briefing(
            window_rallies, window_shots, fps=fps, from_frame=window_start,
            to_frame=to_frame, corners=corners,
        )  # fmt: skip
        briefings.append(b)
        if on_briefing:
            on_briefing(b)
        window_rallies.clear()
        window_shots.clear()
        window_start = to_frame

    for f in range(first, last + 1):
        if speed > 0:
            sleep(1.0 / (fps * speed))
        for p in by_frame.get(f, []):
            detector.push(p)
        handle(detector.poll(f))
        window_rallies.extend(segmenter.advance(f))
        if f - window_start >= interval:
            brief(f)

    handle(detector.flush())
    window_rallies.extend(segmenter.advance(last + 10_000))
    if last - window_start > 0:
        brief(last)
    return briefings
