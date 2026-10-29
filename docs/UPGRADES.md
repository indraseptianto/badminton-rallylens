# Upgrades: Temporal Ensemble & Floor-Gated Rectification

Two accuracy upgrades inspired by the TrackNetV3 temporal-ensemble evaluation
protocol and the height-gated inpaint mask from the badminton analytics
community (concepts, not code — both are documented in
`THIRD_PARTY_NOTICES.md`). This file records *why* the current
implementations look the way they do, so future maintainers don't "simplify"
them back into worse behavior.

## 1. Temporal ensemble in ShuttleTracker

**What:** every frame appears in up to 8 overlapping TrackNet windows (a
9-frame window emits 8 heatmaps; sliding by 1 gives each frame 8 chances to
be detected). Classic TrackNet keeps only the first window's answer.
Ensemble mode collects all window answers and resolves them by **vote
clustering**.

### Voting algorithm (`_emit_from_votes`)

1. Keep every per-window detection above `HEATMAP_THRESH` (0.5) as a vote
   `(x, y, peak)` in inference space.
2. Cluster votes by proximity (≤ 24 px in 512×288 space).
3. The **largest cluster wins**; its members are centroid-averaged weighted
   by heatmap peak; the result is rescaled to original pixels.

### Why voting, not averaging (hard-won lessons)

- **Weighted average of heatmaps → recall collapse.** Averaging dilutes a
  0.9 peak with near-zero noise windows; hard frames (fast shots, small
  shuttle) that only fire in 1-2 windows drop below the 0.5 threshold. On
  the smoke test this cut 22 detections to 5. Max-pooling fixed recall but
  let a single strong false-positive window hijack the location (max error
  1000+ px).
- **Vote clustering gets both:** recall is kept (a frame with a single
  strong window still emits), location is stabilised (one spurious peak is
  outvoted by the consensus cluster), and the output stays deterministic.

### Latency & memory

Frames are only emitted once they can no longer appear in a future window
(i.e. after window `f-1` completes), so results lag by ~8 frames — invisible
for offline batch pipelines, **not** suitable for live streaming. Memory is
bounded: votes are tiny tuples, not full heatmaps (the naive accumulator
held `(H, W)` float arrays per pending frame — ~590 KB each).

**Rule for changes:** if you replace the voting step, run
`tests/test_rectifier.py` AND the smoke E2E (`uv run rallylens detect-shuttle
data/raw/smoke_test.mp4 --no-rectify`) and compare point count against
`--no-ensemble` — recall must not drop.

## 2. Floor-gated rectification in TrajectoryRefiner

**What:** long gaps whose observed endpoints sit in the bottom 5% of the
frame (`height_mask_ratio=0.05`) are *not* filled by the rectifier — they
are dropped from the output.

### Why

A long gap near the floor is the signature of the shuttle having **landed or
left the camera view** (rally over), not of momentary occlusion. Linear
fill / InpaintNet would fabricate a plausible-looking floor trajectory that
never happened — exactly the hallucination the rectifier exists to avoid.
The equivalent height-gating appears in the BadmintonTrackNet inpaint-mask
generation (`th_h = 0.05 · video_height`).

### What is NOT gated

- **Short gaps** (≤ `linear_gap_frames`, default 12) near the floor are
  still linearly filled — a momentary occlusion (net shadow, player leg) is
  real and worth filling.
- **Mid-air long gaps** are still inpainted — that is the primary
  InpaintNet use case (shuttle hidden behind the net).
- Gating only applies when at least **one bounding observed frame** is in
  the floor band; a gap bridging floor → mid-air is gated (conservative).

### Rule for changes

Keep `height_mask_ratio` a constructor param (footage with different camera
angles needs different bands). Before changing the default, re-run the floor
gate tests in `tests/test_upgrades.py` and the E2E smoke test — they encode
the three cases: long floor gap **dropped**, short floor gap **filled**,
mid-air gap **filled**.