"""Shot classification from shuttle trajectory + player detections.

Local, rule-based counterpart of the cloud-hosted shot classifiers used by
other badminton analytics projects (e.g. Roboflow-hosted models). Instead of
calling an external API per frame, we classify each detected ``hit`` event
from kinematics that are already computed locally:

- shuttle speed around the hit (pixels/frame),
- vertical velocity sign (rising vs falling right after the hit),
- hit height relative to the frame (high → net play, low → ground stroke),
- direction (which half of the court the shuttle travels into).

The mapping is deliberately conservative and documented in ``docs/SHOTS.md``;
thresholds are configurable so footage from other cameras can be tuned
without code changes. Runs in pure numpy — no model, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint

# Pixels/frame thresholds tuned on 720p 30fps broadcast footage.
# Convert to the inference space speed the detector already works in:
# events live in original-video pixels, so these are raw-pixel speeds.
SMASH_MIN_SPEED = 30.0   # smash: fastest swing
DRIVE_MIN_SPEED = 18.0   # drive: fast, flat
DROP_MAX_SPEED = 22.0    # drop: slow, falling short
CLEAR_MIN_SPEED = 20.0   # clear/lob: fast + rising
NET_MAX_SPEED = 14.0     # net: very slow, close to the net

# Fraction of frame height marking "high contact" (net/overhead zone).
HIGH_CONTACT_RATIO = 0.35


class ShotEvent(BaseModel):
    """A classified shot (stroke type) associated with a hit event."""

    model_config = ConfigDict(frozen=True)

    frame_idx: int
    player_track_id: int | None
    shot_type: str  # smash | drive | drop | clear | net | unknown
    confidence: float
    speed_px: float
    vy_after: float
    x: int
    y: int


def _speed_between(
    a: ShuttlePoint,
    b: ShuttlePoint,
    fps: float = 30.0,
) -> float:
    """Pixel-per-frame speed between two track points (1 / dt)."""
    dt = b.frame_idx - a.frame_idx
    if dt <= 0:
        return 0.0
    dx = b.x - a.x
    dy = b.y - a.y
    dist = float(np.hypot(dx, dy))
    return dist / dt


def _find_player_near(
    detections: list[Detection],
    frame_idx: int,
    x: int,
    y: int,
    window: int = 15,
    max_dist_px: float = 260.0,
) -> tuple[int, float] | None:
    """Find the (track_id, distance) of the player closest to the hit point.

    Searches ±window frames around the hit; returns the lowest-distance
    candidate whose detection center is within max_dist_px of the hit.
    Returns None if no player is close enough (e.g. referee cam).
    """
    best: tuple[int, float] | None = None
    for d in detections:
        if abs(d.frame_idx - frame_idx) > window:
            continue
        cx = (d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0
        cy = (d.bbox_xyxy[1] + d.bbox_xyxy[3]) / 2.0
        if d.track_id is None:
            continue
        dist = float(np.hypot(cx - x, cy - y))
        if dist <= max_dist_px and (best is None or dist < best[1]):
            best = (d.track_id, dist)
    return best


def classify_hit(
    event: ShuttleEvent,
    track: list[ShuttlePoint],
    detections: list[Detection],
    fps: float = 30.0,
    video_size: tuple[int, int] | None = None,
) -> ShotEvent | None:
    """Classify a single hit event into a stroke type.

    Returns None when the local track lacks enough context around the hit
    (fewer than 2 points on either side) — the caller can skip those.
    """
    frame_idx = event.frame_idx

    before = [p for p in track if p.frame_idx < frame_idx]
    after = [p for p in track if p.frame_idx > frame_idx]
    if len(before) < 2 or len(after) < 2:
        return None

    # Speed approaching and leaving the hit point.
    b1, b2 = before[-2], before[-1]
    a1, a2 = after[0], after[1]
    speed_before = _speed_between(b1, b2, fps)
    speed_after = _speed_between(a1, a2, fps)
    speed = max(speed_before, speed_after)

    # Vertical velocity right after the hit (positive = downward in image
    # coordinates; negative = shuttle is rising).
    vy_after = (a2.y - a1.y) / max(a2.frame_idx - a1.frame_idx, 1)

    # Hit height: use the event y if we have it, else the first after-point.
    y = event.y
    if video_size is None:
        video_size = (int(np.nanmax([p.x for p in track]) + 1), int(np.nanmax([p.y for p in track]) + 1))
    vh = float(video_size[1])
    hit_ratio = y / vh if vh else 0.0
    high_contact = hit_ratio < HIGH_CONTACT_RATIO  # near top of frame

    player = _find_player_near(detections, frame_idx, event.x, event.y)
    player_id = player[0] if player else None

    # --- Rule bank (ordered: most specific first) ---
    if speed <= NET_MAX_SPEED and high_contact:
        shot = ("net", 0.62)
    elif speed >= SMASH_MIN_SPEED and not high_contact and vy_after > 2.0:
        shot = ("smash", 0.85)
    elif speed >= SMASH_MIN_SPEED and vy_after > 0.0:
        shot = ("smash", 0.70)
    elif speed >= DRIVE_MIN_SPEED and abs(vy_after) <= 3.0:
        shot = ("drive", 0.75)
    elif speed >= CLEAR_MIN_SPEED and vy_after < -2.0:
        shot = ("clear", 0.75)
    elif speed <= DROP_MAX_SPEED:
        shot = ("drop", 0.60)
    else:
        shot = ("unknown", 0.40)

    return ShotEvent(
        frame_idx=frame_idx,
        player_track_id=player_id,
        shot_type=shot[0],
        confidence=shot[1],
        speed_px=round(speed, 2),
        vy_after=round(vy_after, 2),
        x=event.x,
        y=event.y,
    )


def classify_shots(
    events: list[ShuttleEvent],
    track: list[ShuttlePoint],
    detections: list[Detection],
    fps: float = 30.0,
    video_size: tuple[int, int] | None = None,
) -> list[ShotEvent]:
    """Classify every ``hit`` event in a rally into a stroke type.

    Non-hit events (landing / out_of_frame) are ignored — they mark rally
    boundaries, not strokes.
    """
    out: list[ShotEvent] = []
    for ev in events:
        if ev.event_type != "hit":
            continue
        shot = classify_hit(ev, track, detections, fps, video_size)
        if shot is not None:
            out.append(shot)
    out.sort(key=lambda s: s.frame_idx)
    return out


__all__ = ["ShotEvent", "classify_shots", "classify_hit"]
