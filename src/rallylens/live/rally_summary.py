"""Per-rally summary and shot history for the live coach view.

Pure: takes a closed ``Rally`` plus the shots/detections observed during it. Anything
that cannot be known (no pose → no player attribution, no calibration → no distance or
landing zone) is reported as ``None`` / ``n/a`` rather than guessed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rallylens.analysis.heatmaps import movement_stats
from rallylens.analysis.rallies import Rally
from rallylens.analysis.shot_classifier import ShotEvent
from rallylens.live.briefing import landing_zone
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.court_geometry import compute_homography, foot_point_from_detection
from rallylens.vision.detect_track import Detection

HISTORY_SHOWN = 12  # shots listed in the text before collapsing to "…"


class ShotLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int  # 1-based within the rally
    frame_idx: int
    player_track_id: int | None
    shot_type: str


class PlayerRally(BaseModel):
    model_config = ConfigDict(frozen=True)

    shots: int
    distance_m: float | None  # None: no calibration or too few tracked frames


class RallySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int  # 1-based over the session
    start_s: float
    duration_s: float
    shot_count: int  # detected hits (>= classified shots)
    history: list[ShotLine]
    players: dict[int, PlayerRally]
    shot_types: dict[str, int]
    landing_zone: str | None  # None: unknown (no landing, no calibration or off court)
    ended_by_landing: bool

    def text(self) -> str:
        head = (
            f"Reli #{self.index} ({self.start_s:.0f}s, {self.duration_s:.1f} dtk, "
            f"{self.shot_count} pukulan)"
        )
        lines = [head]
        if self.players:
            parts = []
            for pid, pr in sorted(self.players.items()):
                dist = "n/a" if pr.distance_m is None else f"{pr.distance_m:.1f} m"
                parts.append(f"pemain #{pid}: {pr.shots} pukulan, {dist}")
            lines.append("  " + " | ".join(parts))
        else:
            lines.append("  Per pemain: n/a (pose belum aktif)")
        if self.history:
            seq = [
                f"{h.number}.{'P' + str(h.player_track_id) + ' ' if h.player_track_id else ''}"
                f"{h.shot_type}"
                for h in self.history[:HISTORY_SHOWN]
            ]
            more = " …" if len(self.history) > HISTORY_SHOWN else ""
            lines.append("  Pukulan: " + " → ".join(seq) + more)
            mix = ", ".join(f"{k} {v}" for k, v in sorted(self.shot_types.items()))
            lines.append(f"  Jenis: {mix}")
        else:
            lines.append("  Jenis pukulan: n/a (klasifikasi belum tersedia)")
        if not self.ended_by_landing:
            lines.append("  Akhir reli: tidak jelas (shuttle hilang)")
        elif self.landing_zone:
            lines.append(f"  Jatuh di zona {self.landing_zone}")
        else:
            lines.append("  Zona jatuh: n/a")
        return "\n".join(lines)


def build_rally_summary(
    rally: Rally,
    index: int,
    shots: Sequence[ShotEvent],
    detections: Sequence[Detection],
    *,
    fps: float,
    corners: CourtCorners | None = None,
) -> RallySummary:
    hit_frames = {h.frame_idx for h in rally.hits}
    mine = sorted((s for s in shots if s.frame_idx in hit_frames), key=lambda s: s.frame_idx)
    history = [
        ShotLine(
            number=i + 1, frame_idx=s.frame_idx, player_track_id=s.player_track_id,
            shot_type=s.shot_type,
        )
        for i, s in enumerate(mine)
    ]  # fmt: skip

    counts = Counter(s.player_track_id for s in mine if s.player_track_id is not None)
    distances: dict[int, float | None] = dict.fromkeys(counts, None)
    if corners is not None and counts:
        h = compute_homography(corners)
        feet: dict[int, list[tuple[int, tuple[int, int]]]] = {pid: [] for pid in counts}
        for d in detections:
            if d.track_id in feet and rally.start_frame <= d.frame_idx <= rally.end_frame:
                pt = foot_point_from_detection(d, h)
                if pt is not None:
                    feet[d.track_id].append((d.frame_idx, pt))
        for pid, seq in feet.items():
            seq.sort(key=lambda t: t[0])
            distances[pid] = movement_stats([f for f, _ in seq], [p for _, p in seq], fps)[0]

    zone = None
    if rally.landing is not None and corners is not None:
        zone = landing_zone(corners, rally.landing.x, rally.landing.y)
    types = Counter(s.shot_type for s in mine if s.shot_type != "unknown")
    return RallySummary(
        index=index,
        start_s=rally.start_frame / fps,
        duration_s=rally.duration_s(fps),
        shot_count=rally.shot_count,
        history=history,
        players={pid: PlayerRally(shots=n, distance_m=distances[pid]) for pid, n in counts.items()},
        shot_types=dict(types),
        landing_zone=zone,
        ended_by_landing=rally.landing is not None,
    )
