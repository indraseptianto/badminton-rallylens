"""Interval briefing for coaches: what happened since the last break, with honest limits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from rallylens.analysis.shot_classifier import ShotEvent
from rallylens.live.segmenter import Rally
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.court_geometry import MARGIN, NET_Y, compute_homography, project_point

MIN_SHOTS_FOR_STATS = 5  # below this, percentages are noise — say so instead
ZONES = ("depan", "tengah", "belakang")  # near net → baseline


class Briefing(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_frame: int
    to_frame: int
    rallies: int
    avg_shots: float | None
    longest_shots: int | None
    longest_s: float | None
    shot_counts: dict[str, int]
    landing_zones: dict[str, int] | None  # None → no court calibration
    low_confidence: bool
    lines: list[str]

    def text(self) -> str:
        return "\n".join(self.lines)


def landing_zone(corners: CourtCorners, x: float, y: float) -> str | None:
    """Depth zone of a landing point; None when it projects off the court."""
    h = compute_homography(corners)
    px, py = project_point(h, x, y)
    cx, cy = px - MARGIN, py - MARGIN
    if not (-60 <= cx <= 670 and -60 <= cy <= 1400):
        return None
    dist = abs(py - (MARGIN + NET_Y)) / NET_Y  # 0 at net → 1 at baseline
    return ZONES[min(2, int(dist * 3))]


def build_briefing(
    rallies: Sequence[Rally],
    shots: Sequence[ShotEvent],
    *,
    fps: float,
    from_frame: int,
    to_frame: int,
    corners: CourtCorners | None = None,
) -> Briefing:
    lines = [f"Ringkasan {from_frame / fps:.0f}s–{to_frame / fps:.0f}s"]
    counts = Counter(s.shot_type for s in shots if s.shot_type != "unknown")
    n_shots = sum(counts.values())

    if not rallies:
        lines.append("Belum ada reli terdeteksi — data belum cukup untuk evaluasi.")
        return Briefing(
            from_frame=from_frame, to_frame=to_frame, rallies=0, avg_shots=None,
            longest_shots=None, longest_s=None, shot_counts=dict(counts),
            landing_zones=None, low_confidence=True, lines=lines,
        )  # fmt: skip

    lengths = [r.shot_count for r in rallies]
    longest = max(rallies, key=lambda r: r.shot_count)
    avg = float(np.mean(lengths))
    lines.append(
        f"{len(rallies)} reli, rata-rata {avg:.1f} pukulan; terpanjang {longest.shot_count} "
        f"pukulan ({longest.duration_s(fps):.0f} dtk)."
    )

    low = n_shots < MIN_SHOTS_FOR_STATS
    if n_shots == 0:
        lines.append("Jenis pukulan: n/a (klasifikasi belum tersedia).")
    elif low:
        listed = ", ".join(f"{k} {v}" for k, v in counts.most_common())
        lines.append(f"Jenis pukulan ({n_shots}, terlalu sedikit untuk persentase): {listed}.")
    else:
        listed = ", ".join(f"{k} {v / n_shots:.0%}" for k, v in counts.most_common())
        lines.append(f"Jenis pukulan ({n_shots}): {listed}.")

    zones: dict[str, int] | None = None
    if corners is None:
        lines.append("Zona jatuh shuttle: n/a (kalibrasi lapangan belum ada).")
    else:
        zones = dict.fromkeys(ZONES, 0)
        for r in rallies:
            if r.landing is not None and (z := landing_zone(corners, r.landing.x, r.landing.y)):
                zones[z] += 1
        total = sum(zones.values())
        if total:
            listed = ", ".join(f"{k} {v}" for k, v in zones.items())
            lines.append(f"Zona jatuh ({total} reli berakhir): {listed}.")
        else:
            zones = None
            lines.append("Zona jatuh: n/a (tidak ada titik jatuh valid).")

    lines.append(
        "Catatan: kecepatan shuttle dari satu kamera hanya perkiraan relatif; "
        "pemain bertukar sisi antar gim."
    )
    return Briefing(
        from_frame=from_frame, to_frame=to_frame, rallies=len(rallies), avg_shots=avg,
        longest_shots=longest.shot_count, longest_s=longest.duration_s(fps),
        shot_counts=dict(counts), landing_zones=zones, low_confidence=low, lines=lines,
    )  # fmt: skip
