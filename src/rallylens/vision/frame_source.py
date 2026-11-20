"""Frame sources for live mode: webcam index, RTSP/HTTP stream, or a video file."""

from __future__ import annotations

from collections.abc import Iterator
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

    def __init__(self, spec: str, fps_fallback: float = DEFAULT_FPS) -> None:
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
                return
            yield idx, frame
            idx += 1
