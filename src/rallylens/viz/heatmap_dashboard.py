"""Six-panel player heatmap dashboard (movement, shots, last shot, endings, types, rally length).

Pure rendering of ``PlayerHeatmap`` data into one BGR image. Panels whose data is missing
say so on the panel instead of drawing an empty court.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from rallylens.analysis.heatmaps import MIN_FOR_STATS, PlayerHeatmap, ShotPoint, key_insights
from rallylens.viz._utils import IMG_H, IMG_W, draw_court_background

W, H = 1536, 1024
BG = (28, 18, 10)
PANEL = (46, 32, 20)
BORDER = (90, 66, 40)
WHITE = (255, 255, 255)
GREY = (170, 160, 150)
CYAN = (255, 200, 60)
FONT = cv2.FONT_HERSHEY_DUPLEX
SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX

PANEL_W, PANEL_H = 487, 396
COURT_W_PX, COURT_H_PX = 404, 202  # court drawn landscape (length horizontal)

# BGR colours per stroke type (legend + dots)
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "smash": (60, 60, 235),
    "clear": (40, 160, 245),
    "drop": (110, 210, 110),
    "drive": (240, 190, 60),
    "net": (230, 90, 150),
}


def _text(
    img: np.ndarray, s: str, org: tuple[int, int], scale: float = 0.5,
    color: tuple[int, int, int] = WHITE, thick: int = 1, font: int = SIMPLEX,
) -> None:  # fmt: skip
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def _court_base() -> np.ndarray:
    """Dark court with white lines, portrait, full diagram resolution."""
    green = draw_court_background()
    lines = (green == 255).all(axis=2)
    base = np.full_like(green, (62, 40, 22))
    base[lines] = (235, 235, 235)
    return base


def _density(
    points: list[tuple[int, int]], weights: list[float] | None = None, sigma: float = 24.0
) -> np.ndarray:
    grid = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for i, (x, y) in enumerate(points):
        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            grid[y, x] += 1.0 if weights is None else weights[i]
    return cv2.GaussianBlur(grid, (0, 0), sigma)


def _heat_on_court(
    base: np.ndarray, grid: np.ndarray, alpha_max: float = 0.8, lo: float = 0.0, hi: float = 1.0
) -> np.ndarray:
    peak = float(grid.max())
    if peak <= 0:
        return base.copy()
    norm = grid / peak
    scaled = np.clip((norm - lo) / max(hi - lo, 1e-6), 0, 1)
    color = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(norm * 1.6, 0, 1)[:, :, None] * alpha_max
    # keep the court lines visible on top of the heat
    lines = np.asarray((base > 200).all(axis=2))[:, :, None]
    out = base.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.where(lines, base, out).astype(np.uint8)


def _to_landscape(img: np.ndarray) -> np.ndarray:
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(rot, (COURT_W_PX, COURT_H_PX), interpolation=cv2.INTER_AREA)


def _dot(img: np.ndarray, x: int, y: int, color: tuple[int, int, int], r: int = 9) -> None:
    cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)


def _colorbar(canvas: np.ndarray, x: int, y: int, top: str, bottom: str) -> None:
    ramp = np.linspace(255, 0, 150).astype(np.uint8)[:, None]
    bar = cv2.applyColorMap(np.repeat(ramp, 14, axis=1), cv2.COLORMAP_JET)
    canvas[y : y + 150, x : x + 14] = bar
    _text(canvas, top, (x - 6, y - 8), 0.38, GREY)
    _text(canvas, bottom, (x - 6, y + 166), 0.38, GREY)


def _panel(
    canvas: np.ndarray, col: int, row: int, n: int, title: str, sub: str,
    court: np.ndarray | None, stats: list[tuple[str, str]], note: str | None = None,
    bar: tuple[str, str] | None = ("High", "Low"),
) -> None:  # fmt: skip
    x0, y0 = 15 + col * (PANEL_W + 11), 130 + row * (PANEL_H + 14)
    cv2.rectangle(canvas, (x0, y0), (x0 + PANEL_W, y0 + PANEL_H), PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + PANEL_W, y0 + PANEL_H), BORDER, 1)
    cv2.circle(canvas, (x0 + 28, y0 + 32), 17, CYAN, -1, cv2.LINE_AA)
    _text(canvas, str(n), (x0 + 22, y0 + 39), 0.6, BG, 2)
    _text(canvas, title, (x0 + 56, y0 + 30), 0.75, WHITE, 1, FONT)
    _text(canvas, sub, (x0 + 56, y0 + 52), 0.42, GREY)
    cy = y0 + 72
    if court is None:
        _text(canvas, note or "n/a", (x0 + 30, cy + 100), 0.5, GREY)
    else:
        canvas[cy : cy + COURT_H_PX, x0 + 12 : x0 + 12 + COURT_W_PX] = court
        if bar:
            _colorbar(canvas, x0 + 12 + COURT_W_PX + 26, cy + 20, *bar)
    sy = y0 + PANEL_H - 62
    cv2.line(canvas, (x0 + 12, sy - 12), (x0 + PANEL_W - 12, sy - 12), BORDER, 1)
    cw = (PANEL_W - 24) // max(len(stats), 1)
    for i, (label, value) in enumerate(stats):
        sx = x0 + 16 + i * cw
        _text(canvas, label, (sx, sy + 12), 0.4, GREY)
        _text(canvas, value, (sx, sy + 44), 0.7, WHITE, 1, FONT)


def _fmt(v: float | None, unit: str, nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f} {unit}"


def _dots_court(
    base: np.ndarray, pts: list[ShotPoint], by_type: bool, grid_pts: list[tuple[int, int]]
) -> np.ndarray:
    heat = _heat_on_court(base, _density(grid_pts, sigma=40.0), alpha_max=0.55)
    for p in pts:
        color = TYPE_COLORS.get(p.shot_type, WHITE) if by_type else (60, 220, 90)
        _dot(heat, p.x, p.y, color)
    return _to_landscape(heat)


def render_heatmap_dashboard(p: PlayerHeatmap, match_label: str) -> np.ndarray:
    """Render one player's dashboard as a BGR image (1536x1024)."""
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    _text(canvas, "Badminton Player", (30, 62), 1.5, WHITE, 3, FONT)
    (tw, _), _ = cv2.getTextSize("Badminton Player ", FONT, 1.5, 3)
    _text(canvas, "Heatmaps", (30 + tw, 62), 1.5, CYAN, 3, FONT)
    _text(canvas, "Player Movement  |  Shot Distribution  |  Performance Insights",
          (32, 100), 0.6, WHITE)  # fmt: skip
    _text(canvas, f"Player: #{p.track_id}", (1130, 60), 0.6, WHITE, 1, FONT)
    _text(canvas, f"Match: {match_label[:24]}", (1130, 90), 0.5, GREY)

    base = _court_base()

    # 1 Movement
    if p.foot_points:
        court = _to_landscape(_heat_on_court(base, _density(p.foot_points)))
        note = None
    else:
        court, note = None, "No tracked feet positions"
    _panel(canvas, 0, 0, 1, "Movement Heatmap", "Where the player moved on court", court,
           [("Total Distance", _fmt(p.distance_m, "m", 0)),
            ("Avg Speed", _fmt(p.avg_speed_ms, "m/s")),
            ("Max Speed (p99)", _fmt(p.max_speed_ms, "m/s"))], note)  # fmt: skip

    # 2 Shot distribution (where the player stood when hitting)
    sp = p.shot_points
    types = Counter(s.shot_type for s in sp)
    if sp:
        court = _dots_court(base, sp, False, [(s.x, s.y) for s in sp])
        note = None
    else:
        court, note = (
            None,
            "No attributed shots (run detect-events, detect-shots, detect --tracker)",
        )
    top = types.most_common(1)[0] if types else None
    _panel(canvas, 1, 0, 2, "Shot Distribution", "Where the player hit from", court,
           [("Total Shots", str(len(sp))),
            ("Top Shot", "n/a" if not top else f"{top[0]} {top[1]}")], note,
           bar=("More", "Fewer"))  # fmt: skip

    # 3 Last shot
    lp = p.last_shot_points
    if lp:
        court = _dots_court(base, lp, False, [(s.x, s.y) for s in lp])
        note = None
    else:
        court, note = None, "No rallies ended by this player"
    lt = Counter(s.shot_type for s in lp).most_common(1)
    _panel(canvas, 2, 0, 3, "Last Shot Heatmap", "Where the player made a rally's last hit",
           court,
           [("Rallies", str(len(lp))),
            ("Last Shot Type", lt[0][0] if lt else "n/a")], note,
           bar=("High", "Low"))  # fmt: skip

    # 4 Rally endings (replaces 'points won': scoring is not available)
    ep = p.ending_points
    if ep:
        court = _to_landscape(_heat_on_court(base, _density(ep, sigma=40.0)))
        note = None
    else:
        court, note = None, "No rally landings detected"
    _panel(canvas, 0, 1, 4, "Rally Ending Heatmap",
           "Where rallies landed after this player's last shot", court,
           [("Landings", str(len(ep))),
            ("Score", "not tracked")], note)  # fmt: skip

    # 5 Shot types
    if sp:
        court = _dots_court(base, sp, True, [])
        note = None
    else:
        court, note = None, "No attributed shots"
    n = sum(types.values())
    legend = [(t, f"{types[t] / n:.0%}" if n >= MIN_FOR_STATS else str(types[t]))
              for t in TYPE_COLORS if types.get(t)]  # fmt: skip
    _panel(canvas, 1, 1, 5, "Shot Type Map", "Each dot is one hit, coloured by stroke", court,
           legend[:3] or [("Shots", "n/a")], note, bar=None)  # fmt: skip
    if sp:  # colour key under the panel title area
        x0, y0 = 15 + 1 * (PANEL_W + 11), 130 + 1 * (PANEL_H + 14)
        for i, (t, c) in enumerate(TYPE_COLORS.items()):
            cv2.circle(canvas, (x0 + 24 + i * 88, y0 + 304), 6, c, -1, cv2.LINE_AA)
            _text(canvas, t, (x0 + 34 + i * 88, y0 + 308), 0.4, GREY)

    # 6 Rally length by court area
    rl = [s for s in sp if s.rally_len is not None]
    if rl:
        pts = [(s.x, s.y) for s in rl]
        wts = [float(s.rally_len or 0) for s in rl]
        num, den = _density(pts, wts, 40.0), _density(pts, None, 40.0)
        mean = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
        lo, hi = float(min(wts)), float(max(wts))
        shown = np.clip((mean - lo) / max(hi - lo, 1e-6), 0, 1)
        color = cv2.applyColorMap((shown * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = np.clip(den / max(float(den.max()), 1e-6) * 1.6, 0, 1)[:, :, None] * 0.8
        lines = np.asarray((base > 200).all(axis=2))[:, :, None]
        mix = base.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
        court = _to_landscape(np.where(lines, base, mix).astype(np.uint8))
        note = None
    else:
        court, note = None, "No rally lengths (needs events + shots)"
    lens = p.rally_lengths
    _panel(canvas, 2, 1, 6, "Rally Length Heatmap", "Avg rally length by where the player hit",
           court,
           [("Avg Rally", _fmt(float(np.mean(lens)) if lens else None, "shots")),
            ("Longest", str(max(lens)) if lens else "n/a"),
            ("Shortest", str(min(lens)) if lens else "n/a")], note,
           bar=("Longer", "Shorter"))  # fmt: skip

    # Footer: insights
    y = 130 + 2 * PANEL_H + 14 + 34
    _text(canvas, "Key Insights:", (30, y), 0.65, CYAN, 1, FONT)
    x = 200
    for line in key_insights(p):
        _text(canvas, line, (x, y), 0.45, WHITE)
        x += 12 + cv2.getTextSize(line, SIMPLEX, 0.45, 1)[0][0] + 24
    _text(canvas, "#id = tracker ID, not a name; players may swap sides between games. "
          "Positions are ground-plane estimates from one camera.", (30, y + 28), 0.4, GREY)  # fmt: skip
    return canvas


def write_heatmap_dashboard(p: PlayerHeatmap, match_label: str, path: Path) -> None:
    """Render and save the dashboard as a PNG."""
    if not cv2.imwrite(str(path), render_heatmap_dashboard(p, match_label)):
        raise RuntimeError(f"could not write {path}")
