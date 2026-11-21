"""Frame sources for live mode: webcam index, RTSP/HTTP stream, or a video file."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class SourceInfo:
    fps: float
    width: int
    height: int


def parse_source(spec: str) -> int | str:
    """``"0"`` → camera index 0; anything else is a URL or path."""
    return int(spec) if spec.isdigit() else spec


class FrameSource:
    """Context manager yielding ``(capture_index, frame)``; index counts frames read."""

    def __init__(
        self,
        spec: str,
        fps_fallback: float = DEFAULT_FPS,
        reconnect_attempts: int = 5,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._spec = spec
        # Files end; only live sources (camera index / URL) are worth reconnecting.
        self._live = spec.isdigit() or "://" in spec
        self._attempts = reconnect_attempts
        self._backoff = backoff_s
        self._sleep = sleep
        self.reconnects = 0
        self._cap = cv2.VideoCapture(parse_source(spec))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open source: {spec}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Cameras/streams often report 0 or nonsense (e.g. 1000); fall back then.
        if not 1.0 <= fps <= 240.0:
            fps = fps_fallback
        self.info = SourceInfo(
            fps=fps,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self._cap.release()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                if self._live and self._reopen():
                    continue
                return
            yield idx, frame
            idx += 1

    def _reopen(self) -> bool:
        """Try to re-open a dropped live source with growing back-off."""
        for attempt in range(1, self._attempts + 1):
            self._sleep(self._backoff * attempt)
            self._cap.release()
            self._cap = cv2.VideoCapture(parse_source(self._spec))
            if self._cap.isOpened():
                self.reconnects += 1
                return True
        return False
