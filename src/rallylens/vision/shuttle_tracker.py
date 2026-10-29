"""Shuttlecock detection using TrackNetV3.

Wraps TrackNet inference with a rolling frame buffer, coordinate extraction
from sigmoid heatmaps (contour-based), and coordinate rescaling to the
original video resolution.

Temporal ensemble
-----------------
Each frame is predicted by every sliding window it appears in (up to
``SEQ_LEN`` windows), the per-frame heatmaps are averaged with a positional
pyramid weight (frames near the window centre are trusted most, mirroring
the TrackNetV3 paper's temporal ensemble), and only then is the coordinate
extracted. This reduces flicker and single-window false positives at almost
no compute cost. Set ``ensemble=False`` to restore the classic one-pass
behaviour.

Weights path: models/shuttle_tracknet.pth
If the weights file is missing, ShuttleTracker logs a warning and
detect() always returns None (graceful fallback).
"""

from __future__ import annotations

import contextlib
import math
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from rallylens.common import get_logger
from rallylens.config import MODELS_DIR
from rallylens.vision.tracknet import TrackNet

_log = get_logger(__name__)

# TrackNetV3 inference resolution
INFER_H = 288
INFER_W = 512

# TrackNetV3 checkpoint: 9 frames → 8 heatmaps
# (first frame in the window has no corresponding output heatmap)
SEQ_LEN = 9  # number of input frames stacked
OUT_LEN = 8  # number of output heatmaps

# Heatmap detection threshold
HEATMAP_THRESH = 0.5

# Default weights filename under MODELS_DIR
DEFAULT_WEIGHTS = "shuttle_tracknet.pth"


def _resolve_device() -> torch.device:
    """Pick the best available torch backend for TrackNet inference.

    Priority: ``RALLYLENS_SHUTTLE_DEVICE`` env var override → MPS (Apple
    Silicon) → CUDA → CPU. The override exists so that users can force CPU
    if an MPS op falls back and slows things down in practice.
    """
    override = os.environ.get("RALLYLENS_SHUTTLE_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ShuttlePoint(BaseModel):
    """Single shuttlecock detection in one frame."""

    model_config = ConfigDict(frozen=True)

    frame_idx: int
    x: int  # pixel x in original video resolution
    y: int  # pixel y in original video resolution


class HitEvent(BaseModel):
    """Shuttle-to-racket contact detected via wrist-proximity heuristic.

    See rallylens.viz._utils.extract_hit_events for the detection algorithm.
    """

    model_config = ConfigDict(frozen=True)

    frame_idx: int
    striker_track_id: int | None
    wrist_distance_px: float
    event_court_xy: tuple[int, int]


def _stack_frames(frames: list[np.ndarray]) -> torch.Tensor:
    """Stack a list of BGR frames into a (1, 3*seq_len, H, W) float tensor."""
    channels = []
    for frame in frames:
        resized = cv2.resize(frame, (INFER_W, INFER_H)).astype(np.float32) / 255.0
        # BGR → RGB channel order
        channels.append(resized[:, :, ::-1].transpose(2, 0, 1))  # (3, H, W)
    stacked = np.concatenate(channels, axis=0)  # (3*seq_len, H, W)
    return torch.from_numpy(stacked).unsqueeze(0)  # (1, 3*seq_len, H, W)


def _ensemble_weights(seq_len: int) -> np.ndarray:
    """Pyramid weights for temporal ensemble: centre frames trusted most.

    For seq_len=8 the pattern is [1, 2, 3, 4, 4, 3, 2, 1] / 20 — the same
    positional weighting used by the TrackNetV3 paper's temporal ensemble.
    """
    weights = np.zeros(seq_len, dtype=np.float64)
    for i in range(math.ceil(seq_len / 2)):
        weights[i] = float(i + 1)
        weights[seq_len - i - 1] = float(i + 1)
    total = float(weights.sum())
    return weights / total


def _predict_location(heatmap: np.ndarray) -> tuple[int, int] | None:
    """Extract (x, y) from a single sigmoid heatmap via contour bounding box.

    Returns the center of the largest contour whose pixel value exceeds the
    threshold, or None if no shuttlecock is detected.
    """
    binary = (heatmap > HEATMAP_THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    return int(bx + bw / 2), int(by + bh / 2)


class ShuttleTracker:
    """Stateful shuttlecock tracker backed by TrackNetV3.

    Usage::

        tracker = ShuttleTracker()
        for idx, frame in enumerate(frames):
            point = tracker.detect(frame, idx)
            if point:
                print(point.x, point.y)
    """

    def __init__(self, weights_path: Path | None = None, ensemble: bool = True) -> None:
        resolved = weights_path or (MODELS_DIR / DEFAULT_WEIGHTS)
        self._model: TrackNet | None = None
        self._enabled = False
        self.ensemble = ensemble

        if not resolved.exists():
            _log.warning(
                "shuttle weights not found at %s — detection disabled. "
                "Provide models/%s to enable.",
                resolved,
                DEFAULT_WEIGHTS,
            )
            return

        self._device = _resolve_device()
        model = TrackNet(in_dim=3 * SEQ_LEN, out_dim=OUT_LEN)
        with contextlib.chdir(MODELS_DIR):
            state = torch.load(resolved, map_location=self._device)
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        self._model = model
        self._enabled = True
        _log.info("shuttle TrackNet loaded from %s (device=%s)", resolved, self._device)

        self._buffer: deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        # frame_idx corresponding to each buffer slot (oldest → newest)
        self._idx_buffer: deque[int] = deque(maxlen=SEQ_LEN)
        # Temporal ensemble state: per-frame detection votes — list of
        # (x, y, heatmap_peak) from every overlapping window. Keyed by
        # frame_idx.
        self._vote_acc: dict[int, list[tuple[int, int, float]]] = {}
        self._win_count = 0  # number of completed inference windows
        # Original frame size of the most recent frame (used by ensemble
        # extraction); updated on every detect() call.
        self._last_frame_size = (0, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[ShuttlePoint]:
        """Add a frame to the buffer and run detection when buffer is full.

        Args:
            frame:     BGR numpy array (any resolution).
            frame_idx: 0-based frame index in the source video.

        Returns:
            List of ShuttlePoint — one per detected frame in the current
            window (up to SEQ_LEN points), or an empty list if the buffer
            is not yet full or the model is disabled.
        """
        if not self._enabled or self._model is None:
            return []

        self._last_frame_size = frame.shape[:2]
        orig_h, orig_w = frame.shape[:2]
        self._buffer.append(frame)
        self._idx_buffer.append(frame_idx)

        if len(self._buffer) < SEQ_LEN:
            return []

        tensor = _stack_frames(list(self._buffer))
        if self._device.type != "cpu":
            tensor = tensor.to(self._device)

        with torch.no_grad():
            output = self._model(tensor)  # (1, SEQ_LEN, H, W)
        output_cpu = output.detach().cpu()
        self._win_count += 1

        if self.ensemble:
            return self._detect_ensemble(output_cpu)

        # Classic one-pass mode: emit each heatmap immediately.
        points: list[ShuttlePoint] = []
        for i in range(OUT_LEN):
            heatmap = output_cpu[0, i].numpy()  # (H, W) float in [0, 1]
            xy = _predict_location(heatmap)
            if xy is None:
                continue
            x = int(xy[0] * self._last_frame_size[1] / INFER_W)
            y = int(xy[1] * self._last_frame_size[0] / INFER_H)
            points.append(ShuttlePoint(frame_idx=self._idx_buffer[i + 1], x=x, y=y))
        return points

    def flush(self) -> list[ShuttlePoint]:
        """Emit any remaining ensemble-accumulated frames (end of video).

        In ensemble mode some trailing frames are still pending when the
        video ends because they may not have been fully observed by all
        windows. Call this once after the last frame to collect them.
        """
        if not self.ensemble or not self._enabled:
            return []
        points: list[ShuttlePoint] = []
        for f in sorted(self._vote_acc):
            emitted = self._emit_from_votes(f, self._vote_acc[f])
            if emitted is not None:
                points.append(emitted)
        self._vote_acc.clear()
        return points

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_ensemble(self, output_cpu: torch.Tensor) -> list[ShuttlePoint]:
        """Temporal-ensemble branch: vote on per-window peaks, then emit
        every frame that can no longer appear in a future window.

        For each frame we keep the detected (x, y, peak) from every window
        it appears in. On emission the detections are clustered by proximity;
        the weighted centroid of the largest cluster is the ensemble answer.
        This keeps recall (a frame detected in a single window still counts)
        while stabilising the location (a spurious peak in one window is
        outvoted by the cluster) — unlike heatmap averaging, which dilutes
        the 0.9 peak below the 0.5 threshold and drops hard frames.
        """
        # Heatmap i corresponds to buffer slot i+1.
        for i in range(OUT_LEN):
            frame_idx = self._idx_buffer[i + 1]
            heatmap = output_cpu[0, i].numpy()  # (H, W) float in [0, 1]
            xy = _predict_location(heatmap)
            if xy is None:
                continue
            peak = float(heatmap.max())
            votes = self._vote_acc.setdefault(frame_idx, [])
            votes.append((xy[0], xy[1], peak))

        # A frame at index f is seen by windows that start in [f-8, f-1]:
        # it is fully observed once window f-1 has completed (win_count == f).
        done = [f for f in self._vote_acc if f < self._win_count]
        emitted: list[ShuttlePoint] = []
        for f in sorted(done):
            point = self._emit_from_votes(f, self._vote_acc.pop(f))
            if point is not None:
                emitted.append(point)
        return emitted

    def _emit_from_votes(
        self,
        frame_idx: int,
        votes: list[tuple[int, int, float]],
    ) -> ShuttlePoint | None:
        """Resolve per-window detection votes into one ensembled point."""
        orig_h, orig_w = self._last_frame_size
        if orig_w == 0 or orig_h == 0:
            return None

        # Cluster votes: group by proximity; the majority cluster wins
        # (a spurious single-window peak is outvoted even if it is strong),
        # then weight-average the survivors by their heatmap peak.
        strong = [v for v in votes if v[2] >= HEATMAP_THRESH]
        if not strong:
            return None
        strong.sort(key=lambda v: v[2], reverse=True)

        clusters: list[list[tuple[int, int, float]]] = []
        for v in strong:
            anchor_x, anchor_y = v[0], v[1]
            placed = False
            for cl in clusters:
                ax, ay = cl[0][0], cl[0][1]
                if abs(anchor_x - ax) <= 24 and abs(anchor_y - ay) <= 24:
                    cl.append(v)
                    placed = True
                    break
            if not placed:
                clusters.append([v])

        best = max(clusters, key=len)
        wsum = sum(v[2] for v in best)
        if wsum <= 0:
            return None
        cx = sum(v[0] * v[2] for v in best) / wsum
        cy = sum(v[1] * v[2] for v in best) / wsum

        return ShuttlePoint(
            frame_idx=frame_idx,
            x=int(cx * orig_w / INFER_W),
            y=int(cy * orig_h / INFER_H),
        )
