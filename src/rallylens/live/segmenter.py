"""Group shuttle events into rallies (first hit → landing / long silence)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rallylens.analysis.events.detector import ShuttleEvent


class Rally(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_frame: int
    end_frame: int
    hits: list[ShuttleEvent]
    landing: ShuttleEvent | None  # None → ended by silence, outcome unknown

    @property
    def shot_count(self) -> int:
        return len(self.hits)

    def duration_s(self, fps: float) -> float:
        return (self.end_frame - self.start_frame) / fps


class RallySegmenter:
    """Feed events in frame order; closed rallies come back from ``push`` / ``advance``."""

    def __init__(self, fps: float, max_gap_s: float = 4.0, min_hits: int = 2) -> None:
        self._max_gap = max(1, round(max_gap_s * fps))
        self._min_hits = min_hits
        self._hits: list[ShuttleEvent] = []

    def _close(self, end: int, landing: ShuttleEvent | None) -> list[Rally]:
        hits, self._hits = self._hits, []
        if len(hits) < self._min_hits:
            return []  # a lone "hit" is a serve fragment or false positive
        return [Rally(start_frame=hits[0].frame_idx, end_frame=end, hits=hits, landing=landing)]

    def push(self, ev: ShuttleEvent) -> list[Rally]:
        closed: list[Rally] = []
        if self._hits and ev.frame_idx - self._hits[-1].frame_idx > self._max_gap:
            closed += self._close(self._hits[-1].frame_idx, None)
        if ev.event_type == "hit":
            self._hits.append(ev)
        elif ev.event_type == "landing" and self._hits:
            closed += self._close(ev.frame_idx, ev)
        return closed

    def advance(self, current_frame: int) -> list[Rally]:
        """Close a rally whose last hit is older than the allowed gap."""
        if self._hits and current_frame - self._hits[-1].frame_idx > self._max_gap:
            return self._close(self._hits[-1].frame_idx, None)
        return []
