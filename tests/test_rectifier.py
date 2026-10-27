"""Tests for trajectory rectification (TrackNetV3 InpaintNet + gap-fill)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rallylens.vision.rectifier import TrajectoryRefiner, load_inpaintnet_weights
from rallylens.vision.shuttle_tracker import ShuttlePoint
from rallylens.vision.tracknet import InpaintNet

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def refiner():
    """Refiner with real InpaintNet weights (skip silently if missing)."""
    w = MODELS / "inpaintnet_best.pth"
    if not w.exists():
        pytest.skip("models/inpaintnet_best.pth not present")
    return TrajectoryRefiner(w, linear_gap_frames=3)


def _trajectory(n: int = 60, gap: tuple[int, int] | None = None, spread: bool = False):
    """Diagonal trajectory (x,y) = (bg+i*3, 160+i*2), optional hole.

    ``spread=True`` uses a realistic mid-screen trajectory; the default keeps
    the tiny corner drift used by the linear-only tests.
    """
    bg = 300 if spread else 10
    pts = []
    for i in range(n):
        if gap and gap[0] <= i < gap[1]:
            continue
        pts.append(ShuttlePoint(frame_idx=i, x=bg + i * 3, y=(160 if spread else 100) + i * 2))
    return pts


def test_load_weights_remap(refiner):
    """Official checkpoint key remapping loads into the local InpaintNet."""
    model = InpaintNet()
    state = load_inpaintnet_weights(MODELS / "inpaintnet_best.pth", "cpu")
    model.load_state_dict(state)
    out = model(
        torch.zeros((1, 16, 2)),
        torch.zeros((1, 16, 1)),
    )
    assert out.shape == (1, 16, 2)


def test_no_gap_no_change(refiner):
    """A continuous track passes through untouched."""
    pts = _trajectory()
    out = refiner.refine(pts, video_size=(1280, 720))
    assert len(out) == len(pts)
    assert all(p.frame_idx == q.frame_idx for p, q in zip(pts, out, strict=False))


def test_short_gap_linear_fill(refiner):
    """Gaps <= linear_gap_frames are filled by linear interpolation."""
    pts = _trajectory(gap=(20, 23))  # 3 missing frames (idx 20..22)
    out = refiner.refine(pts, video_size=(1280, 720))
    assert len(out) == 60
    # Filled point at frame 21 should sit between 20 and 23.
    by_frame = {p.frame_idx: p for p in out}
    assert 21 in by_frame
    # x = 10 + 21*3 = 73 → within a couple pixels of the true line.
    assert abs(by_frame[21].x - 73) <= 3
    assert abs(by_frame[21].y - 142) <= 3


def test_long_gap_inpaint(refiner):
    """Long mid-clip gaps get InpaintNet rectification near the true line."""
    n = 60
    gap = (20, 32)  # 12 missing frames (true occlusion: shuttle behind net)
    pts = _trajectory(n=n, gap=gap, spread=True)
    out = refiner.refine(pts, video_size=(1280, 720))
    by_frame = {p.frame_idx: p for p in out}
    filled = [i for i in range(*gap) if i in by_frame]
    assert len(filled) >= 8, "expected InpaintNet to fill most of the gap"
    # Edge frames of the gap are the most constrained → tight tolerance.
    for i in (gap[0], gap[0] + 1, gap[1] - 1):
        assert abs(by_frame[i].x - (300 + i * 3)) <= 30, (
            f"frame {i} far off: {by_frame[i].x} vs {300 + i * 3}"
        )
    # Interior frames should at least land in the right half-plane of the court.
    for i in filled:
        assert by_frame[i].x > 300, f"frame {i} rectified outside court: {by_frame[i].x}"


def test_empty_input(refiner):
    assert refiner.refine([]) == []


def test_single_point(refiner):
    out = refiner.refine([ShuttlePoint(frame_idx=5, x=100, y=200)], video_size=(1280, 720))
    assert len(out) == 1
