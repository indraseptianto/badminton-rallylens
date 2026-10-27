# Rally Event Detection (landing / hit / out_of_frame)

Rule-based event detector ported from
[ZSHYC/BadmintonTrackNet](https://github.com/ZSHYC/BadmintonTrackNet)
(`tracknet/events/`), adapted for RallyLens coordinate conventions and coding
standards.

## What it does

Turns the *repaired shuttle trajectory* (from `detect-shuttle`) into a list of
rally events:

| Event | Meaning | Primary rules used |
|---|---|---|
| `landing` | shuttle hits the court floor | `speed_step`, `low_speed_start`, `speed_drop`, `trajectory_end` |
| `hit` | racket strikes the shuttle | `vy_reversal`, `vx_reversal`, `acceleration_peak` |
| `out_of_frame` | shuttle leaves the visible image | `visibility_drop_edge`, `visibility_drop_upward`, `visibility_drop_high_altitude` |

Detection is **two-phase** in the original project:

1. **Phase 1 (rule bank, always on)** — `KinematicsCalculator` computes
   velocity / acceleration / curvature; `BounceCandidateGenerator` runs 7
   primary + 2 auxiliary rules per frame, merges nearby candidates by rule
   priority, and flags trajectory-end landings. Pure numpy, no GPU, no
   learned weights.
2. **Phase 2 (BounceNet, NOT ported)** — a small classifier that filters
   rule false-positives using trajectory features and/or visual patches.
   Requires a trained checkpoint (`tracknet.events.bouncenet` in the source
   repo, trained with `tracknet.training.bouncenet`) — see "Enabling
   BounceNet later" below.

## Files

| File | Contents |
|---|---|
| `src/rallylens/analysis/events/kinematics.py` | velocity / acceleration / curvature (fps-aware) |
| `src/rallylens/analysis/events/rules.py` | `BounceCandidateGenerator`, `RallyBounceDetector` (rule bank) |
| `src/rallylens/analysis/events/detector.py` | `BounceDetector` facade + `ShuttleEvent` pydantic model |
| `src/rallylens/pipeline/events.py` | pipeline stage `run_events_pipeline` |
| `tests/test_event_detection.py` | 8 tests (kinematics, rules, merge, sparse input, empty input) |

## Coordinate convention (IMPORTANT)

The original rule thresholds were tuned in **TrackNet model space
(512×288)** — that is why `BounceDetector` internally rescales the input
trajectory into 512×288 before running the rule bank, and maps detected event
coordinates **back to original-video pixels** in the returned
`ShuttleEvent`. Pass coordinates in original pixels and you never have to
think about the rescale.

Kinematics velocity is expressed in pixels/frame normalised to a 30 fps
reference (same convention as the source project). If your clip is not 30
fps, pass `fps=` to `BounceDetector` / `run_events_pipeline` so the scale
factor stays correct.

## Usage

```bash
# needs a shuttle track first (data/tracks/<id>/<id>_shuttle.jsonl)
uv run rallylens detect-events path/to/video.mp4

# or programmatically
from rallylens.pipeline.events import run_events_pipeline
events = run_events_pipeline(video_path, video_id, fps=30)
for e in events:
    print(e.frame_idx, e.event_type, e.rule, round(e.confidence, 2), e.x, e.y)
```

Output artifact: `data/events/<video_id>/<video_id>_events.jsonl` — a list of
`ShuttleEvent`:

```json
{"frame_idx": 33, "x": 212, "y": 645, "event_type": "hit",
 "rule": "vy_reversal", "confidence": 0.9, "all_rules": ["vy_reversal"],
 "features": {"vy_before": 12.1, "vy_after": -11.3, ...}}
```

The full pipeline (`rallylens run`) runs event detection automatically after
shuttle detection; a failure there never blocks the rest of the run
(best-effort stage).

## Threshold tuning

All thresholds are constructor params of `BounceDetector` →

| Param | Default | Rule it gates |
|---|---|---|
| `speed_drop_ratio` | 0.3 | landing: speed drops below 30% of prior |
| `min_speed_before_landing` | 8.0 | landing: must be moving before the drop |
| `max_speed_after_landing` | 5.0 | landing: must be (near) stopped after |
| `min_speed_at_hit` | 5.0 | hit: shuttle must still be fast after impact |
| `vy_reversal_threshold` | 3.0 | hit: vertical velocity sign flip |
| `acc_threshold` | 3.0 | hit: acceleration spike height |
| `merge_window` | 3 | frames within which candidates collapse to the best rule |
| `edge_margin` | 20 | out_of_frame: distance from image border (model space) |

Defaults come from the source project (tuned on broadcast matches). If your
footage is amateur/phone camera with a lot of motion blur, expect more
false `hit`s — raise `min_speed_at_hit` / `vy_reversal_threshold` first.

## Enabling BounceNet later

Phase 2 is intentionally left out: it needs a trained checkpoint, an extra
torch dependency surface, and (in visual mode) frame patch extraction —
none of which earn their keep on a CPU-only VPS with no training data yet.

To add it when you have a checkpoint:

1. Port `tracknet/events/bouncenet.py` + `visual_features.py` from
   ZSHYC/BadmintonTrackNet into `analysis/events/`.
2. Train the checkpoint with their `tracknet.training.bouncenet` on your own
   labeled events (the semi-auto labeling tool in the same repo is the
   intended label source).
3. Re-add the `bouncenet_ckpt` kwarg to `BounceDetector` and call
   `predictor.classify_candidates(...)` after `generate(...)` (see their
   `detector.py::detect`).

## Migrating event files (Schema)

`ShuttleEvent` is a frozen pydantic model serialized as JSONL, mirroring the
`ShuttlePoint` convention. Adding fields later is backward-compatible for
readers (extra keys ignored), but old files without new fields will fail at
the pydantic level — bump the model with defaults when you extend it.