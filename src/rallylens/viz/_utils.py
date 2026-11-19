"""Internal shared helpers for the viz package.

Court diagram constants, homography computation, and foot-position extraction
are centralised here so overlay.py and viz_court.py stay thin.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

import cv2
import numpy as np

from rallylens.vision.court_geometry import (
    COURT_H,
    COURT_W,
    MARGIN,
    NET_Y,
    extract_hit_events,
)
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint

# Gaussian blur kernel width: (sigma * multiplier) + 1
_BLUR_KERNEL_MULTIPLIER: Final[int] = 4

# Fading-trail radius scaling
_TRAIL_RADIUS_MIN_FACTOR: Final[float] = 0.4
_TRAIL_RADIUS_MAX_FACTOR: Final[float] = 0.6
_TRAIL_RADIUS_MIN_PX: Final[int] = 2

# ---------------------------------------------------------------------------
# Court diagram constants (1 px = 1 cm)
# ---------------------------------------------------------------------------


IMG_W: int = COURT_W + 2 * MARGIN  # 730
IMG_H: int = COURT_H + 2 * MARGIN  # 1460

# Y-offsets from the top of the court (cm / px)
_NET_Y: int = NET_Y
_SHORT_SVC_FAR: int = _NET_Y - 198  # 472
_SHORT_SVC_NEAR: int = _NET_Y + 198  # 868
_DBL_LONG_FAR: int = 76
_DBL_LONG_NEAR: int = COURT_H - 76  # 1264

# X-offsets
_SINGLES_LEFT: int = 46
_SINGLES_RIGHT: int = COURT_W - 46  # 564
_CENTER_X: int = COURT_W // 2  # 305

# ---------------------------------------------------------------------------
# Track-color palette (BGR)
# ---------------------------------------------------------------------------

TRACK_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (0, 128, 255),  # sky-blue / orange
    (255, 128, 0),  # orange-blue
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 255, 0),  # yellow
]

# Shared shuttle highlight color (BGR: yellow)
SHUTTLE_COLOR: tuple[int, int, int] = (0, 255, 255)


def track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


# ---------------------------------------------------------------------------
# Court diagram drawing
# ---------------------------------------------------------------------------


def draw_court_background() -> np.ndarray:
    """Return a (IMG_H, IMG_W, 3) BGR image of a standard badminton court."""
    img: np.ndarray = np.full((IMG_H, IMG_W, 3), (34, 85, 34), dtype=np.uint8)
    m = MARGIN
    white = (255, 255, 255)

    def hline(y: int) -> None:
        cv2.line(img, (m, m + y), (m + COURT_W, m + y), white, 2)

    def vline(x: int) -> None:
        cv2.line(img, (m + x, m), (m + x, m + COURT_H), white, 2)

    # Outer doubles boundary
    cv2.rectangle(img, (m, m), (m + COURT_W, m + COURT_H), white, 2)
    # Singles sidelines
    vline(_SINGLES_LEFT)
    vline(_SINGLES_RIGHT)
    # Net
    hline(_NET_Y)
    # Short service lines
    hline(_SHORT_SVC_FAR)
    hline(_SHORT_SVC_NEAR)
    # Doubles long service lines
    hline(_DBL_LONG_FAR)
    hline(_DBL_LONG_NEAR)
    # Center service line (full court length for clarity)
    vline(_CENTER_X)

    return img


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------


def compute_shuttle_court_positions(
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    H: np.ndarray,
    *,
    kp_conf_thresh: float = 0.3,
    hit_radius_factor: float = 0.9,
) -> dict[int, tuple[int, int]]:
    """Compute per-frame top-down shuttle positions via TRACE's hit+interp trick.

    Thin composition: calls `extract_hit_events` for the event list and then
    linearly interpolates between consecutive events so the trail draws
    cleanly. See `extract_hit_events` for the hit-detection details.

    Returns a dict mapping frame_idx -> (x, y) in court-diagram pixel space.
    Empty dict if no hit events are detected.
    """
    hits = extract_hit_events(
        detections,
        shuttle_track,
        H,
        kp_conf_thresh=kp_conf_thresh,
        hit_radius_factor=hit_radius_factor,
    )
    if not hits:
        return {}

    result: dict[int, tuple[int, int]] = {}
    for i in range(len(hits) - 1):
        t0, p0 = hits[i].frame_idx, hits[i].event_court_xy
        t1, p1 = hits[i + 1].frame_idx, hits[i + 1].event_court_xy
        dt = t1 - t0
        if dt <= 0:
            result[t0] = p0
            continue
        for j in range(dt):
            alpha = j / dt
            x = int(round(p0[0] + alpha * (p1[0] - p0[0])))
            y = int(round(p0[1] + alpha * (p1[1] - p0[1])))
            result[t0 + j] = (x, y)
    result[hits[-1].frame_idx] = hits[-1].event_court_xy

    return result


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------


def build_heatmap_over_court(
    court: np.ndarray,
    positions: list[tuple[int, int]],
    *,
    blur_sigma: int = 12,
    alpha_max: float = 0.75,
) -> np.ndarray:
    """Return a copy of `court` with a Gaussian-smoothed JET heatmap of `positions` blended on.

    Positions outside the image bounds are silently skipped. If positions is
    empty (or all out of bounds so the accumulated grid is zero) a plain copy
    of `court` is returned.
    """
    if not positions:
        return court.copy()

    grid: np.ndarray = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for x, y in positions:
        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            grid[y, x] += 1.0

    kernel_size = blur_sigma * _BLUR_KERNEL_MULTIPLIER + 1
    grid = cv2.GaussianBlur(grid, (kernel_size, kernel_size), blur_sigma)

    max_val = float(grid.max())
    if max_val == 0:
        return court.copy()

    normalized = (grid / max_val * 255).astype(np.uint8)
    heatmap_rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    alpha_map = (grid / max_val * alpha_max).astype(np.float32)[:, :, np.newaxis]
    blended = (
        court.astype(np.float32) * (1 - alpha_map) + heatmap_rgb.astype(np.float32) * alpha_map
    ).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# Fading-trail rendering
# ---------------------------------------------------------------------------


def draw_fading_trail(
    frame: np.ndarray,
    points: Iterable[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    head_radius: int,
) -> None:
    """Draw a chronological point trail: newest point is largest + brightest.

    Each point is rendered as a filled circle whose BGR channels and radius
    scale linearly with its position in the trail, so older points fade toward
    black and shrink. Works on any backdrop (plain court, heatmap, or raw
    video frame) because it fades toward black instead of toward a fixed
    background color.
    """
    pts = list(points)
    n = len(pts)
    if n == 0:
        return
    for i, pt in enumerate(pts):
        alpha = (i + 1) / n
        radius = max(
            _TRAIL_RADIUS_MIN_PX,
            int(head_radius * (_TRAIL_RADIUS_MIN_FACTOR + _TRAIL_RADIUS_MAX_FACTOR * alpha)),
        )
        faded = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.circle(frame, pt, radius, faded, -1, cv2.LINE_AA)


def render_pip_court_frame(
    court_bg: np.ndarray,
    player_trails: Mapping[int, Iterable[tuple[int, int]]],
    shuttle_trail: Iterable[tuple[int, int]],
    *,
    player_radius: int = 6,
    shuttle_radius: int = 4,
) -> np.ndarray:
    """Render a single court-diagram frame with player and shuttle trails."""
    frame = court_bg.copy()

    for tid, trail in player_trails.items():
        color = track_color(tid)
        pts = list(trail)
        draw_fading_trail(frame, pts, color=color, head_radius=player_radius)
        if pts:
            cv2.circle(frame, pts[-1], player_radius, color, -1, cv2.LINE_AA)

    s_pts = list(shuttle_trail)
    draw_fading_trail(frame, s_pts, color=SHUTTLE_COLOR, head_radius=shuttle_radius)
    if s_pts:
        cv2.circle(frame, s_pts[-1], shuttle_radius, SHUTTLE_COLOR, -1, cv2.LINE_AA)

    return frame
