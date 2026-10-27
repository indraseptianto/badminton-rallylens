"""Badminton event detection (rule-based Phase 1) — ported from
ZSHYC/BadmintonTrackNet, adapted for rallylens coordinate conventions.

The original project detects events in TrackNet's model space (512×288) with
thresholds tuned at that resolution. RallyLens shuttle tracks live in
original-video pixels (720p/1080p), so this detector rescales the trajectory
into 512×288 before running the rule bank and maps event coordinates back to
the original image. No learning model is required — pure numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from rallylens.analysis.events.kinematics import KinematicsCalculator
from rallylens.analysis.events.rules import BounceCandidateGenerator

# TrackNetV3 model input size — the resolution the rule thresholds were tuned at.
MODEL_W = 512
MODEL_H = 288


class ShuttleEvent(BaseModel):
    """A detected badminton rally event (landing / hit / out_of_frame)."""

    model_config = ConfigDict(frozen=True)

    frame_idx: int
    x: int  # pixels in original video resolution
    y: int  # pixels in original video resolution
    event_type: str  # 'landing' | 'hit' | 'out_of_frame'
    rule: str
    confidence: float
    all_rules: list[str]
    features: dict[str, Any]


class BounceDetector:
    """Detect landing / hit / out_of_frame events from a shuttle trajectory.

    Coordinates are accepted and returned in **original video pixels**;
    internally they are rescaled into the 512×288 model space where the rule
    thresholds were designed.
    """

    def __init__(
        self,
        fps: int = 30,
        # landing parameters
        speed_drop_ratio: float = 0.3,
        min_speed_before_landing: float = 8.0,
        max_speed_after_landing: float = 5.0,
        # hit parameters
        min_speed_at_hit: float = 5.0,
        vy_reversal_threshold: float = 3.0,
        acc_threshold: float = 3.0,
        # local-extrema parameters
        min_height_diff: float = 5.0,
        min_peak_speed: float = 10.0,
        min_speed_diff: float = 2.0,
        # general
        min_visible_before: int = 3,
        merge_window: int = 3,
        edge_margin: int = 20,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.kinematics_calculator = KinematicsCalculator(fps=fps)
        self.candidate_generator = BounceCandidateGenerator(
            speed_drop_ratio=speed_drop_ratio,
            min_speed_before_landing=min_speed_before_landing,
            max_speed_after_landing=max_speed_after_landing,
            min_speed_at_hit=min_speed_at_hit,
            vy_reversal_threshold=vy_reversal_threshold,
            acc_threshold=acc_threshold,
            min_height_diff=min_height_diff,
            min_peak_speed=min_peak_speed,
            min_speed_diff=min_speed_diff,
            min_visible_before=min_visible_before,
            merge_window=merge_window,
            edge_margin=edge_margin,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        x: np.ndarray,
        y: np.ndarray,
        visibility: np.ndarray,
        frame_ids: np.ndarray | None = None,
    ) -> list[ShuttleEvent]:
        """Detect events from a trajectory in original-video pixels.

        Args:
            x, y: coordinate arrays (original pixel space), same length.
            visibility: 1=observed, 0=occluded/missing.
            frame_ids: real frame indices; defaults to ``arange(len(x))``.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        visibility = np.asarray(visibility, dtype=np.float64)
        if len(x) != len(y) or len(x) != len(visibility):
            raise ValueError("x, y and visibility must have the same length")
        if len(x) == 0:
            return []
        if frame_ids is None:
            frame_ids = np.arange(len(x))
        frame_ids_arr = np.asarray(frame_ids)
        if len(frame_ids_arr) != len(x):
            raise ValueError("frame_ids must match the trajectory length")

        # Original image size (from observed coords) for mapping back.
        w = float(np.nanmax(x) + 1)
        h = float(np.nanmax(y) + 1)

        # Rescale into model space where thresholds were tuned.
        x_m = x * (MODEL_W / w)
        y_m = y * (MODEL_H / h)

        kinematics = self.kinematics_calculator.compute(x_m, y_m, visibility)
        candidates = self.candidate_generator.generate(
            x_m,
            y_m,
            visibility,
            kinematics=kinematics,
            img_height=MODEL_H,
            img_width=MODEL_W,
        )

        events: list[ShuttleEvent] = []
        for cand in candidates:
            pos = int(cand["frame"])
            events.append(
                ShuttleEvent(
                    frame_idx=int(frame_ids_arr[pos]),
                    x=int(round(cand["x"] * (w / MODEL_W))),
                    y=int(round(cand["y"] * (h / MODEL_H))),
                    event_type=cand["event_type"],
                    rule=cand["rule"],
                    confidence=float(cand["confidence"]),
                    all_rules=list(cand.get("all_rules", [])),
                    features=dict(cand.get("features", {})),
                )
            )
        return events

    def detect_from_shuttle_points(
        self,
        frame_idxs: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> list[ShuttleEvent]:
        """Detect events from sparse per-frame observations.

        Frames without a detection are treated as occluded (visibility=0),
        which is exactly the repair-mask semantics used by the rectifier.
        """
        if len(frame_idxs) == 0:
            return []
        frame_idxs = np.asarray(frame_idxs)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        min_f = int(frame_idxs.min())
        max_f = int(frame_idxs.max())
        span = max_f - min_f + 1
        dense_x = np.full(span, np.nan, dtype=np.float64)
        dense_y = np.full(span, np.nan, dtype=np.float64)
        dense_x[frame_idxs - min_f] = xs
        dense_y[frame_idxs - min_f] = ys

        # Interpolate interior holes so kinematics stay defined, but mark
        # original gaps as invisible for the rule bank.
        valid = np.where(~np.isnan(dense_x))[0]
        visibility = np.zeros(span, dtype=np.float64)
        visibility[valid] = 1.0
        if len(valid) > 1:
            dense_x = np.interp(np.arange(span), valid, dense_x[valid])
            dense_y = np.interp(np.arange(span), valid, dense_y[valid])
        dense_x = np.nan_to_num(dense_x)
        dense_y = np.nan_to_num(dense_y)

        return self.detect(
            dense_x,
            dense_y,
            visibility,
            frame_ids=np.arange(min_f, max_f + 1),
        )


__all__ = ["BounceDetector", "ShuttleEvent"]
