"""Camera loop: a reader thread keeps the freshest frames, the consumer runs the models."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

import numpy as np

from rallylens.live.queue import DropOldestQueue
from rallylens.live.session import LiveEngine
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint

Detect = Callable[[np.ndarray, int], list[ShuttlePoint]]
Pose = Callable[[np.ndarray, int], list[Detection]]


def run_live(
    frames: Iterable[tuple[int, np.ndarray]],
    detect: Detect,
    engine: LiveEngine,
    *,
    queue_size: int = 8,
    max_frames: int | None = None,
    flush: Callable[[], list[ShuttlePoint]] | None = None,
    pose: Pose | None = None,
) -> int:
    """Pump frames through ``detect`` into ``engine``; returns frames dropped.

    Capture indices keep counting even when frames are dropped, so the engine's
    clock (frame_idx / fps) stays truthful when the models cannot keep up.
    """
    q: DropOldestQueue[tuple[int, np.ndarray]] = DropOldestQueue(queue_size)
    done = threading.Event()
    stop = threading.Event()

    def reader() -> None:
        try:
            for n, item in enumerate(frames):
                if stop.is_set() or (max_frames is not None and n >= max_frames):
                    break
                q.put(item)
        finally:
            done.set()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    last = 0
    try:
        while True:
            item = q.get(timeout=0.05)
            if item is None:
                if done.is_set() and len(q) == 0:
                    break
                continue
            idx, frame = item
            last = idx
            engine.feed(idx, detect(frame, idx), pose(frame, idx) if pose else ())
    finally:
        stop.set()
        t.join(timeout=2.0)
    if flush is not None:
        engine.feed(last, flush())
    return q.dropped
