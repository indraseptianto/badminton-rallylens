"""Per-player spatial statistics on the court: movement, shot positions, rally endings.

Pure data (no rendering). Court-diagram coordinates are 1 px = 1 cm including the
``MARGIN`` border, the same space as ``viz`` uses. Everything that needs a floor
position goes through the player's *feet* (a valid floor-plane projection); aerial
shuttle pixels are never projected as ground positions, except rally landings, which
are floor contacts by definition.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from rallylens.analysis.events.detector import ShuttleEvent
from rallylens.analysis.rallies import RallySegmenter
from rallylens.analysis.shot_classifier import ShotEvent
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.court_geometry import (
    MARGIN,
    NET_Y,
    compute_homography,
    foot_point_from_detection,
    project_point,
)
from rallylens.vision.detect_track import Detection

SMOOTH_WINDOW = 5  # frames; damps keypoint jitter that would inflate distance
MAX_GAP_FRAMES = 3  # steps across longer detection gaps are not counted
MAX_SPEED_MS = 10.0  # faster than any badminton footwork → tracking glitch
SHOT_LOOKUP_FRAMES = 5  # foot position is taken from the nearest frame within this window
MIN_FOR_STATS = 5  # fewer samples than this → "n/a" rather than a percentage


class ShotPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    shot_type: str
    rally_len: int | None  # shots in the rally this shot belonged to, if known


class PlayerHeatmap(BaseModel):
    model_config = ConfigDict(frozen=True)

    track_id: int
    foot_points: list[tuple[int, int]]
    distance_m: float | None
    avg_speed_ms: float | None
    max_speed_ms: float | None  # 99th percentile of valid steps, not the raw maximum
    shot_points: list[ShotPoint]
    last_shot_points: list[ShotPoint]
    ending_points: list[tuple[int, int]]  # where rallies landed after this player's last shot
    rally_lengths: list[int]  # lengths of rallies this player took part in


def zone_of(y_px: float) -> str:
    """Depth zone ('front'/'mid'/'back') of a court-diagram y relative to the net."""
    dist = abs(y_px - (MARGIN + NET_Y)) / NET_Y
    return ("front", "mid", "back")[min(2, int(dist * 3))]


def _movement(
    frames: list[int], pts: list[tuple[int, int]], fps: float
) -> tuple[float | None, float | None, float | None]:
    if len(pts) < SMOOTH_WINDOW + 1:
        return None, None, None
    arr = np.asarray(pts, dtype=float)
    k = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    # 'valid' convolution trims edges instead of biasing them toward zero
    sx = np.convolve(arr[:, 0], k, mode="valid")
    sy = np.convolve(arr[:, 1], k, mode="valid")
    fr = np.asarray(frames[SMOOTH_WINDOW // 2 : SMOOTH_WINDOW // 2 + len(sx)])
    gaps = np.diff(fr)
    step_cm = np.hypot(np.diff(sx), np.diff(sy))
    ok = (gaps > 0) & (gaps <= MAX_GAP_FRAMES)
    speeds = np.zeros_like(step_cm)
    speeds[ok] = step_cm[ok] / 100.0 / (gaps[ok] / fps)
    ok &= speeds <= MAX_SPEED_MS
    if not ok.any():
        return None, None, None
    dist_m = float(step_cm[ok].sum() / 100.0)
    time_s = float(gaps[ok].sum() / fps)
    return dist_m, dist_m / time_s, float(np.percentile(speeds[ok], 99))


def build_player_heatmaps(
    detections: Sequence[Detection],
    shots: Sequence[ShotEvent],
    events: Sequence[ShuttleEvent],
    corners: CourtCorners,
    fps: float,
    *,
    top_n: int = 2,
) -> list[PlayerHeatmap]:
    """Spatial statistics for the ``top_n`` most frequently tracked players."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    h = compute_homography(corners)
    counts = Counter(d.track_id for d in detections if d.track_id is not None)
    ids = [tid for tid, _ in counts.most_common(top_n)]

    feet: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for d in sorted(detections, key=lambda d: d.frame_idx):
        if d.track_id in ids:
            pt = foot_point_from_detection(d, h)
            if pt is not None and d.track_id is not None:
                feet[d.track_id].append((d.frame_idx, pt))

    def foot_near(tid: int, frame: int) -> tuple[int, int] | None:
        seq = feet.get(tid, [])
        i = bisect.bisect_left(seq, frame, key=lambda t: t[0])
        cands = [seq[j] for j in (i - 1, i) if 0 <= j < len(seq)]
        cands = [c for c in cands if abs(c[0] - frame) <= SHOT_LOOKUP_FRAMES]
        return min(cands, key=lambda c: abs(c[0] - frame))[1] if cands else None

    # Rallies from events (same segmentation the live briefing uses).
    seg = RallySegmenter(fps)
    rallies = []
    for ev in sorted(events, key=lambda e: e.frame_idx):
        rallies += seg.push(ev)
    if events:
        rallies += seg.advance(max(e.frame_idx for e in events) + 10_000)
    shot_by_frame = {s.frame_idx: s for s in shots}
    rally_len_by_frame: dict[int, int] = {}
    last_hit_frames: dict[int, int] = {}  # last-hit frame → index into rallies
    for i, r in enumerate(rallies):
        for hit in r.hits:
            rally_len_by_frame[hit.frame_idx] = r.shot_count
        last_hit_frames[r.hits[-1].frame_idx] = i

    out: list[PlayerHeatmap] = []
    for tid in ids:
        seq = feet.get(tid, [])
        dist, avg, mx = _movement([f for f, _ in seq], [p for _, p in seq], fps)
        shot_pts: list[ShotPoint] = []
        last_pts: list[ShotPoint] = []
        endings: list[tuple[int, int]] = []
        lengths: list[int] = []
        for s in shots:
            if s.player_track_id != tid or s.shot_type == "unknown":
                continue
            pos = foot_near(tid, s.frame_idx)
            if pos is None:
                continue
            sp = ShotPoint(
                x=pos[0], y=pos[1], shot_type=s.shot_type,
                rally_len=rally_len_by_frame.get(s.frame_idx),
            )  # fmt: skip
            shot_pts.append(sp)
            if (ri := last_hit_frames.get(s.frame_idx)) is not None:
                last_pts.append(sp)
                landing = rallies[ri].landing
                if landing is not None:
                    endings.append(project_point(h, landing.x, landing.y))
        for r in rallies:
            if any(
                (sh := shot_by_frame.get(hit.frame_idx)) and sh.player_track_id == tid
                for hit in r.hits
            ):
                lengths.append(r.shot_count)
        out.append(
            PlayerHeatmap(
                track_id=tid, foot_points=[p for _, p in seq], distance_m=dist,
                avg_speed_ms=avg, max_speed_ms=mx, shot_points=shot_pts,
                last_shot_points=last_pts, ending_points=endings, rally_lengths=lengths,
            )
        )  # fmt: skip
    return out


def key_insights(p: PlayerHeatmap) -> list[str]:
    """Plain-language observations, each backed by enough samples or marked n/a."""
    lines: list[str] = []
    if len(p.foot_points) >= 30:
        zc = Counter(zone_of(y) for _, y in p.foot_points)
        zone, n = zc.most_common(1)[0]
        lines.append(f"Most time spent in the {zone} court ({n / len(p.foot_points):.0%}).")
    else:
        lines.append("Movement: not enough tracked frames for a zone summary.")
    types = Counter(s.shot_type for s in p.shot_points)
    if sum(types.values()) >= MIN_FOR_STATS:
        t, n = types.most_common(1)[0]
        lines.append(
            f"Most used shot: {t} ({n / sum(types.values()):.0%} of {sum(types.values())})."
        )
    else:
        lines.append("Shots: too few classified shots for percentages.")
    if len(p.rally_lengths) >= 3:
        lines.append(f"Average rally length: {float(np.mean(p.rally_lengths)):.1f} shots.")
    else:
        lines.append("Rallies: too few to summarise.")
    return lines
