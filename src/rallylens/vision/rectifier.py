"""Trajectory rectification for shuttlecock tracks — TrackNetV3 InpaintNet.

TrackNetV3 (ACM MM 2023): after the TrackNet heatmap stage predicts a
shuttlecock trajectory, occluded / uncertain frames are repaired by a 1D U-Net
(InpaintNet) that inpaints the missing coordinates conditioned on a binary
mask. This module implements that two-stage refinement on top of the sparse
``ShuttlePoint`` list produced by ``ShuttleTracker``.

Architecture & weights: qaz812345/TrackNetV3, MIT License,
Copyright (c) 2024 qaz812345 — see THIRD_PARTY_NOTICES.md.

Refinement pipeline
-------------------
1. Build a dense per-frame coordinate series from the sparse detections.
2. Short gaps (<= ``linear_gap_frames``) are filled with linear interpolation
   (cheap, no model needed — this covers the common "shuttle hidden for a
   couple of frames behind the net" case).
3. Remaining gaps are inpainted by InpaintNet with a sliding window of
   ``seq_len`` frames. A window is only used when it holds enough observed
   frames (>= ``min_valid_ratio``); otherwise the gap is left as-is, because
   long gaps usually mean "no rally" rather than "occluded shuttle".

Weights: ``models/inpaintnet_best.pth`` (from the TrackNetV3 repo checkpoints).
The official checkpoint stores the trainer state under ``ckpt["model"]`` and
misspells the bottleneck keys as ``buttleneck.*`` — both are handled here.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import torch

from rallylens.config import MODELS_DIR
from rallylens.vision.shuttle_tracker import ShuttlePoint
from rallylens.vision.tracknet import InpaintNet

_log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = "inpaintnet_best.pth"

# InpaintNet was trained with sequence length 16.
SEQ_LEN = 16
# Sliding window stride (50% overlap).
STRIDE = SEQ_LEN // 2
# Mask ratio seen during training is 0.3; require at least this much real
# observation per window so we never inpaint fully-empty windows.
MIN_VALID_RATIO = 0.4


def load_inpaintnet_weights(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load a TrackNetV3 InpaintNet checkpoint and remap its keys.

    The official ``InpaintNet_best.pt`` is a full trainer checkpoint
    (``{"epoch", "model", ...}``) whose state dict uses ``buttleneck.conv_1`` /
    ``buttleneck.conv_2`` — a typo for the ``bottleneck`` block our local
    ``InpaintNet`` class names ``bottleneck.block.0`` / ``bottleneck.block.1``.
    """
    with contextlib.chdir(MODELS_DIR):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    remapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        new_key = (
            key.replace("buttleneck.conv_1", "bottleneck.block.0")
            .replace("buttleneck.conv_2", "bottleneck.block.1")
        )
        remapped[new_key] = value
    return remapped


class TrajectoryRefiner:
    """Fill gaps and rectify a shuttlecock trajectory with InpaintNet.

    Usage::

        refiner = TrajectoryRefiner()
        refined = refiner.refine(points)   # list[ShuttlePoint]

    If the InpaintNet weights are missing, the refiner degrades gracefully:
    short gaps still get linear interpolation, long gaps are untouched.
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        seq_len: int = SEQ_LEN,
        linear_gap_frames: int = 12,
    ) -> None:
        resolved = weights_path or (MODELS_DIR / DEFAULT_WEIGHTS)
        self.seq_len = seq_len
        self.linear_gap_frames = linear_gap_frames
        self._model: InpaintNet | None = None
        self._enabled = False

        if not resolved.exists():
            _log.warning(
                "inpaint weights not found at %s — rectification disabled "
                "(linear gap-fill still active). Provide models/%s to enable.",
                resolved,
                DEFAULT_WEIGHTS,
            )
            return

        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")

        model = InpaintNet()
        state = load_inpaintnet_weights(resolved, self._device)
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        self._model = model
        self._enabled = True
        _log.info("InpaintNet loaded from %s (device=%s)", resolved, self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self,
        points: list[ShuttlePoint],
        video_size: tuple[int, int] | None = None,
    ) -> list[ShuttlePoint]:
        """Rectify a sparse shuttle track into a denser, smoother one.

        Args:
            points: Raw detections from :class:`ShuttleTracker`. Coordinates
                are expected in original-video pixel space.
            video_size: ``(width, height)`` of the source video, used to
                normalise coordinates into InpaintNet's ``[0, 1]`` space.
                If omitted, normalisation falls back to the max observed
                coordinate per axis (accurate for 16:9 sources).

        Returns a new list of ShuttlePoint covering frame range
        [min_frame_idx, max_frame_idx]; frames that could not be estimated
        with confidence (long empty windows) are omitted.
        """
        if not points:
            return []

        # 1. Dense series indexed by frame_idx → (x, y) or NaN.
        frame_idxs = np.array([p.frame_idx for p in points])
        xs = np.array([p.x for p in points], dtype=np.float64)
        ys = np.array([p.y for p in points], dtype=np.float64)

        min_f = int(frame_idxs.min())
        max_f = int(frame_idxs.max())
        span = max_f - min_f + 1
        dense = np.full((span, 2), np.nan, dtype=np.float64)
        dense[frame_idxs - min_f] = np.stack([xs, ys], axis=1)

        # Remember which frames were originally occluded/missing: those are the
        # ones InpaintNet is asked to rectify.
        missing: np.ndarray = np.asarray(np.isnan(dense).any(axis=1), dtype=bool)

        # 2. Seed gaps with linear interpolation (short gaps = final answer,
        # long gaps = plausible input for InpaintNet). Half-inpaint: seed
        # every frame, then let the model refine only the occluded ones.
        dense = self._fill_gaps_linear(dense)

        # 3. InpaintNet rectification on the seeded trajectory. Only gaps
        # longer than ``linear_gap_frames`` go through the model; short gaps
        # are already answered by the cheap linear fill.
        if self._enabled and self._model is not None:
            missing_rectify = self._long_gap_mask(missing)
            dense = self._inpaint_gaps(dense, missing_rectify, video_size)

        # 4. Emit ShuttlePoint for every frame that got an estimate.
        # Frames in windows too sparsely observed (e.g. between rallies) are
        # dropped rather than hallucinated.
        out: list[ShuttlePoint] = []
        for offset in range(span):
            row = dense[offset]
            if np.isnan(row).any():
                continue
            out.append(ShuttlePoint(frame_idx=min_f + offset, x=int(round(row[0])), y=int(round(row[1]))))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _long_gap_mask(self, missing: np.ndarray) -> np.ndarray:
        """Keep only missing frames that belong to a run longer than
        ``linear_gap_frames``; short runs are left to linear fill."""
        n = len(missing)
        out = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if not missing[i]:
                i += 1
                continue
            j = i
            while j < n and missing[j]:
                j += 1
            if (j - i) > self.linear_gap_frames:
                out[i:j] = True
            i = j
        return out

    def _fill_gaps_linear(self, dense: np.ndarray) -> np.ndarray:
        """Linearly interpolate every gap between observed frames.

        Also fills leading/trailing NaN runs with the nearest observed value
        so the trajectory covers the full clip evenly.
        """
        coords = dense[:, 0]
        # Find observed indices.
        valid = np.where(~np.isnan(coords))[0]
        if len(valid) == 0:
            return dense
        # Fill interior gaps.
        for a, b in zip(valid[:-1], valid[1:], strict=False):
            gap_len = b - a - 1
            if gap_len <= 0:
                continue
            x0, y0 = dense[a]
            x1, y1 = dense[b]
            for k in range(1, gap_len + 1):
                t = k / (gap_len + 1)
                dense[a + k] = [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]
        # Fill head/tail with the nearest observed frame.
        dense[: valid[0]] = dense[valid[0]]
        dense[valid[-1] + 1 :] = dense[valid[-1]]
        return dense

    def _inpaint_gaps(
        self,
        dense: np.ndarray,
        missing: np.ndarray,
        video_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Rectify originally-missing frames with InpaintNet sliding windows.

        Only windows with enough observed context (>= MIN_VALID_RATIO) are
        used; frames inside sparser windows are set back to NaN so they get
        dropped from the output (they are inter-rally gaps, not occlusion).
        """
        assert self._model is not None
        n = len(dense)
        # Work in normalized [0,1] space like the training setup. ShuttlePoint
        # coords are in original-video pixels; normalise against the source
        # video size so InpaintNet sees true ratios regardless of aspect.
        vw, vh = video_size or (int(np.nanmax(dense[:, 0]) + 1), int(np.nanmax(dense[:, 1]) + 1))
        norm = dense.copy()
        norm[:, 0] = norm[:, 0] / vw
        norm[:, 1] = norm[:, 1] / vh

        # Frames we choose NOT to trust → NaN → dropped at emit time.
        drop = np.zeros(n, dtype=bool)
        refined_any = np.zeros(n, dtype=bool)

        for start in range(0, n, STRIDE):
            end = min(start + self.seq_len, n)
            if end - start < 4:
                break
            win_missing = missing[start:end]
            if not win_missing.any():
                continue
            if win_missing.sum() / (end - start) > (1.0 - MIN_VALID_RATIO):
                # Too little observation — candidate for dropping, unless a
                # neighbouring window successfully rectifies those frames.
                drop[start:end] |= win_missing
                continue

            coords = norm[start:end].copy()  # (L, 2) observed coords
            coords[win_missing] = 0.0        # (L, 2) zero-mask, like training
            mask = win_missing.astype(np.float32).reshape(-1, 1)  # (L, 1)
            x_t = torch.from_numpy(coords.astype(np.float32)).unsqueeze(0).to(self._device)  # (1, L, 2)
            m_t = torch.from_numpy(mask).unsqueeze(0).to(self._device)    # (1, L, 1)

            with torch.no_grad():
                out = self._model(x_t, m_t)  # (1, L, 2) in [0, 1]

            refined = out.detach().cpu().numpy()[0]  # (L, 2)
            for k in range(len(win_missing)):
                if win_missing[k]:
                    norm[start + k, 0] = refined[k, 0]
                    norm[start + k, 1] = refined[k, 1]
                    refined_any[start + k] = True

        # Back to pixel space. Drop only frames no window managed to rectify.
        dense[:, 0] = norm[:, 0] * vw
        dense[:, 1] = norm[:, 1] * vh
        dense[drop & ~refined_any] = np.nan
        return dense
