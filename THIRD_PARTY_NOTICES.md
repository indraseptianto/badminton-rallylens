# Third-Party Notices & Attribution

This repository builds on and adapts substantial code from open-source
badminton computer-vision projects. We are grateful to their authors. Per the
MIT License terms, the copyright notices below are reproduced and every
adapted component is mapped to its origin so future maintainers can trace any
conflict or update upstream.

## 1. TrackNetV3 — shuttlecock tracking model & weights

| | |
|---|---|
| **Project** | [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) (paper: *TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification*, ACM Multimedia Asia 2023) |
| **License** | [MIT](https://github.com/qaz812345/TrackNetV3/blob/master/LICENSE) — Copyright (c) 2024 qaz812345 |
| **Adapted here** | `docs/TRACKNETV3.md` (integration guide) · `src/rallylens/vision/tracknet.py`, `rectifier.py`, `shuttle_tracker.py` (port/integration of the two-stage TrackNet + InpaintNet architecture) |
| **Weights shipped** | `models/shuttle_tracknet.pth`, `models/inpaintnet_best.pth` (from the official `TrackNetV3_ckpts.zip` release — explicitly released under MIT by the project) |

The rectification input protocol (zero-masked coordinates) and the
`buttleneck.*` → `bottleneck.block.*` key remap were reverse-engineered
against the official checkpoints; both are documented in `docs/TRACKNETV3.md`.

## 2. BadmintonTrackNet — rally event detection (rule bank + BounceNet)

| | |
|---|---|
| **Project** | [ZSHYC/BadmintonTrackNet](https://github.com/ZSHYC/BadmintonTrackNet) — TrackNetV3-based shuttlecock tracking and event analysis |
| **License** | MIT — the repo retains the original project license, Copyright (c) 2024 qaz812345 (see its `LICENSE`) |
| **Adapted here** | `src/rallylens/analysis/events/` — `kinematics.py`, `rules.py` (BounceCandidateGenerator), `detector.py` (BounceDetector facade) |
| **Upstream files** | `tracknet/events/kinematics.py` · `tracknet/events/candidate_generator.py` · `tracknet/events/detector.py` (in the ZSHYC repo) |
| **Not ported** | `tracknet/events/bouncenet.py`, `visual_features.py`, `dataset.py`, training code (Phase 2 classifier — needs a trained checkpoint + GPU; see `docs/EVENTS.md` for how to add it back) |

## 3. BadmintonVision pipeline (original RallyLens lineage)

| | |
|---|---|
| **Project** | [YeonSeong-Lee/rallylens](https://github.com/YeonSeong-Lee/rallylens) — original RallyLens pipeline (player detection with YOLO11-pose, court calibration, visualization) |
| **License** | Declared MIT in its `pyproject.toml` (no LICENSE file upstream at time of writing) |
| **Adapted here** | the overall pipeline structure: `src/rallylens/pipeline/`, `vision/detect_track.py`, `vision/court_detector.py`, `viz/`, `ingest/`, CLI skeleton |

## License summary

| Component in this repo | License | Copyright |
|---|---|---|
| Original RallyLens pipeline + adaptations | MIT | 2026 RallyLens contributors |
| TrackNetV3 integration + shipped weights | MIT | 2024 qaz812345 |
| Event detection rule bank (analysis/events) | MIT | 2024 qaz812345 |
| `web/` (Vite app) | AGPL-3.0-only | 2026 RallyLens contributors (see `web/LICENSE`) |

## How to update this file

Whenever you port or vendor code from another project:

1. Add a row with the exact upstream repo + file paths + license + copyright.
2. Reproduce the upstream copyright notice inside the adapted source file
   header (MIT requires it).
3. Point the corresponding `docs/*.md` guide at the upstream reference.
4. If the upstream license differs from MIT, note it in the License summary
   and keep the two codebases cleanly separable.