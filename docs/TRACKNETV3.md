# TrackNetV3 Integration & Model-Switching Guide

RallyLens tracks the shuttlecock with **TrackNetV3** (two-stage: heatmap
detection + trajectory rectification), then applies a model-free linear
gap-fill. This document explains how the pipeline is wired, how the official
checkpoints are loaded, and how to swap in a different model later.

## Architecture

```
video frames
   │  (per frame, sliding window of 9)
   ▼
ShuttleTracker (TrackNet — 2D U-Net heatmap)
   │  sparse ShuttlePoint list: frame_idx, x, y   (only confident detections)
   ▼
TrajectoryRefiner (optional, default ON)
   │  1. linear gap-fill (gaps ≤ linear_gap_frames, default 12 frames)
   │  2. InpaintNet (1D U-Net) rectifies longer occlusion gaps
   ▼
dense, smooth shuttle track  (saved to data/tracks/<id>/<id>_shuttle.jsonl)
```

Sources:

- `src/rallylens/vision/tracknet.py`    — model architecture (TrackNet + InpaintNet)
- `src/rallylens/vision/shuttle_tracker.py` — per-frame detector wrapper
- `src/rallylens/vision/rectifier.py`   — TrajectoryRefiner (NEW: gap-fill + InpaintNet)
- `src/rallylens/pipeline/shuttle.py`   — pipeline stage, wires the two together

## Models

| File | Purpose | Source | Size |
|---|---|---|---|
| `models/shuttle_tracknet.pth` | raw heatmap detector | [TrackNetV3 repo ckpts](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing) | 45 MB |
| `models/inpaintnet_best.pth` | trajectory rectifier | same zip (`InpaintNet_best.pt`) | 6.3 MB |

### Key remapping (PORTANT when you re-download weights)

The official `InpaintNet_best.pt` is a full trainer checkpoint, not a bare
state dict:

```python
ckpt = torch.load(path, weights_only=True)   # keys: epoch, model, optimizer, scheduler
raw = ckpt["model"]
```

Its keys also contain a **misspelling** (`buttleneck.conv_1` / `buttleneck.conv_2`)
that must be remapped to our local class names (`bottleneck.block.0` /
`bottleneck.block.1`). This is handled by `load_inpaintnet_weights` in
`rectifier.py` — if your new checkpoint fails `load_state_dict`, check the key
names first and extend the remap dict.

### Rectification protocol (matches the paper's training setup)

Official InpaintNet was trained with **zero-masked coordinates**: the frames
to repair get `(x, y) = (0, 0)` in the input and a mask of `1`; observed
frames keep their real values and mask `0`. Do NOT seed missing frames with
linear interpolations as model input — the network is sensitive to that and
produces garbage. Empirically verified on the official checkpoint:

| Input strategy for masked frames | Result |
|---|---|
| linear-seeded coords + mask | ❌ model drifts, 100s of px error |
| **zero-masked coords + mask** | ✅ error ≈ 3–30 px on synthetic 12-frame gap |

Only the model output for masked frames is used; observed frames keep their
real detections. Windows with < 40 % observed frames are treated as
"no rally" (inter-point gaps between rallies) and left missing rather than
hallucinated.

### Input resolution & coordinate space

- TrackNet inference happens at **288 × 512 (H × W)**; `ShuttleTracker`
  rescales detected coordinates back to the *original video resolution*.
- `TrajectoryRefiner` therefore normalises against the original video size
  (`video_size` arg) so InpaintNet sees true aspect ratios in `[0, 1]`.

## Running

```bash
uv run rallylens detect-shuttle path/to/video.mp4                 # rectified (default)
uv run rallylens detect-shuttle path/to/video.mp4 --no-rectify   # raw detections only
```

Log line on success:

```
rectification: 22 raw → 28 refined points (+6)
```

## Tests

```bash
uv run pytest tests/test_rectifier.py -v
```

Unit tests run the real checkpoint against synthetic trajectories (straight
diagonal mid-court motion) and assert: no-gap passthrough, linear fill for
short gaps, InpaintNet rectification for long gaps, and graceful empty
input. Skipped automatically when `models/inpaintnet_best.pth` is absent.

## Switching to a different model later

1. **Drop-in replacement for TrackNet** — keep the `ShuttleTracker` interface
   (`detect(frame, frame_idx) -> list[ShuttlePoint]`). Any detector that
   outputs per-frame pixel coordinates works: YOLO fine-tuned on shuttlecock,
   FloMatch, etc. Just update `_predict_location` / the heatmap decode if the
   new model emits boxes instead of heatmaps.
2. **Drop-in replacement for InpaintNet** — keep the signature
   `(coords: (N, L, 2) normalized [0,1], mask: (N, L, 1) binary) -> refined (N, L, 2)`.
   That is exactly the paper's interface, so the official `TrackNetV3` repo
   weights drop in with only the key remap above.
3. **Retraining** — the official repo (`qaz812345/TrackNetV3`) trains both
   stages on the Shuttlecock Trajectory Dataset; see its `train.py` /
   `generate_mask_data.py`. For venue-specific lighting use the labeling tool
   linked in that repo. On a CPU-only VPS expect ~minutes per minute of video;
   training itself needs a GPU (Colab works).
4. **Graceful degradation is built in** — if `inpaintnet_best.pth` is missing,
   the refiner logs a warning and falls back to pure linear gap-fill. If
   `shuttle_tracknet.pth` is missing, `ShuttleTracker` disables cleanly and
   the pipeline continues with player tracking only.

## Migrating old tracks (Schema)

`data/tracks/<id>/<id>_shuttle.jsonl` stays a list of
`{"frame_idx": int, "x": int, "y": int}` — rectification only adds more
entries (denser), it never changes the shape, so downstream metrics /
visualizations need no changes.