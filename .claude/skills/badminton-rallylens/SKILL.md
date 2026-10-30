---
name: badminton-rallylens
description: Use when working on RallyLens badminton vision AI pipeline. TrackNetV3 shuttle tracking, event detection, shot classification, CLI, tests, and debugging.
---

# Badminton RallyLens — Vision AI Pipeline

RallyLens turns a badminton match video (local file or YouTube URL) into
player tracks (YOLO11-pose + ByteTrack), shuttlecock tracks (TrackNetV3 with
temporal ensemble + InpaintNet rectification), rally events
(landing/hit/out_of_frame), shot types (smash/drive/drop/clear/net), court
calibration, and visualizations. CPU-only friendly; Python 3.11+, `uv`-managed.

**Read the docs first** — deep guides live in `docs/`:
- `docs/TRACKNETV3.md` — TrackNetV3 + InpaintNet protocol (zero-mask input,
  key remap). Read before touching `vision/tracknet.py`,
  `vision/shuttle_tracker.py`, `vision/rectifier.py`.
- `docs/EVENTS.md` — event rule bank, coordinate convention, BounceNet extension.
- `docs/SHOTS.md` — shot classification thresholds & tuning.
- `docs/UPGRADES.md` — temporal ensemble + floor-gate design rationale
  (**read before "simplifying" the ensemble — it was already debugged**).
- `THIRD_PARTY_NOTICES.md` — attribution map. Mandatory reading when adapting
  external code.

## Setup

```bash
cd /opt/badminton-dashboard/rallylens
uv sync                      # install deps; ffmpeg must be on PATH
PYTHONPATH= uv run rallylens --help    # NOTE: PYTHONPATH= prefix needed in
                                       # Hermes terminal (site-packages shadowing)
```

Weights are committed: `models/shuttle_tracknet.pth`, `models/inpaintnet_best.pth`
(need `git lfs` or plain files, already in-commit). YOLO downloads on first run.

## CLI & artifact map

| Command | Writes |
|---|---|
| `ingest <yt-url>` | `data/raw/<video_id>.mp4` (supports `--start/--end` clips) |
| `detect <video>` | `data/detections/<id>/<id>_players.jsonl` |
| `detect-shuttle <video>` | `data/tracks/<id>/<id>_shuttle.jsonl`; flags `--no-rectify`, `--no-ensemble` |
| `detect-events <video>` | `data/events/<id>/<id>_events.jsonl` |
| `detect-shots <video>` | `data/shots/<id>/<id>_shots.jsonl` (needs track+events) |
| `calibrate <video>` | `data/calibration/<id>/corners.json` (`--interactive` picker) |
| `viz <video>` | `data/viz/<id>/` overlay MP4 + court GIF |
| `run <url\|path>` | one-shot: everything incl. events+shots (best-effort) |

`video_id` = input file stem. Every clip owns `data/<kind>/<video_id>/`.

## Architecture layers (respect the boundaries)

`config` (paths only) → `serialization` (pydantic JSON/JSONL) →
`pipeline.io` (only place mapping video_id → artifact paths + typed save/load)
→ `vision` (models: `detect_track`, `tracknet`, `shuttle_tracker`,
`rectifier`, `court_detector`) → `analysis` (pure-numpy: `events/`,
`shot_classifier`, `metrics`) → `pipeline` (thin orchestration) →
`viz` (rendering, no inference) → `cli` (click shim).

- Only `vision` may import `ultralytics`/`cv2`. New framework-tangled code
  stays in already-relaxed modules (`vision.detect_track`,
  `analysis.events.rules`) — don't spread mypy relaxations.
- New artifact type → add path + save/load in `pipeline.io`, not at callers.
- `orchestrator.run_full_pipeline` runs events + shots after shuttle
  detection; both are best-effort (`try/except`, never block tracking).

## Conventions & invariants

- Pydantic v2, `frozen=True` for detection records. mypy strict except
  `tool.mypy.overrides` in pyproject. Ruff: `E,F,I,W,B,UP,SIM`, 100 cols.
- **Coordinates**: detector works internally in 512×288 model space and
  rescales back to original pixels — never convert manually.
- **Control flow**: `run` = events best-effort but `detect-shuttle` failure
  stops (`st.stop` in dashboard too).
- Attribution: any ported code MUST update `THIRD_PARTY_NOTICES.md` AND
  reproduce the upstream MIT copyright notice in the source header.
- **CLAUDE.md is user-consent-protected**: do not patch it without explicit
  approval; route docs to `docs/*.md` instead.

## Debugging playbook (hard-won lessons)

1. **Temporal ensemble = vote clustering, NOT averaging/max-pooling.**
   Averaging dilutes the 0.9 heatmap peak below 0.5 on hard frames (recall
   collapse: 22→5 points on smoke test); max-pooling lets a single
   false-positive window win (1000+ px error). Current impl collects
   `(x, y, peak)` votes per frame from every overlapping window, clusters by
   proximity, majority cluster's peak-weighted centroid wins. Deterministic.
2. **InpaintNet input protocol**: masked frames set to **0.0** (zero-mask),
   mask=1 marks frames to refine; coords normalized by original video size;
   only masked frames' outputs are used. Checkpoint keys remap
   `buttleneck.*` → `bottleneck.block.*`; checkpoint is trainer state → take
   `ck["model"]`.
3. **`_last_frame_size` must be set in `detect()`** (`frame.shape[:2]`) or
   ensemble emits 0 points — the setter was once lost in a full-file rewrite.
4. **Floor gate**: long gaps whose bounding observed frames lie in the bottom
   5% of the frame = landing/off-view → drop, never inpaint (anti-hallucination).
   Short gaps (≤12 frames) and mid-air gaps still fill.
5. **Ports from ZSHYC/BadmintonTrackNet**: `rules.py` keeps Chinese docstrings
   + attribution header (traceability); mypy override exists because the
   rule bank is dynamic dict literals. `frame_ids` may be None → `np.asarray`
   before subscript.
6. **Orchestrator**: dummy/synthetic videos make `read_video_properties` fail
   → try/except `fps=30` fallback; don't reference `props` after.
7. **`write_file` refuses to overwrite files it hasn't read** — `read_file`
   first, then rewrite (hit on first detector.py attempt).
8. **Run separaately long E2E commands**; chained detect-shuttle runs in one
   shell command time out (exit 124) on CPU.
9. Ruff B905: `zip(a, b, strict=False)` is required by the config.

## Regression checks (always run after touching vision/analysis)

```bash
cd /opt/badminton-dashboard/rallylens
PYTHONPATH= uv run pytest tests/ -q          # expect all green (68 tests)
PYTHONPATH= uv run ruff check src/ tests/
PYTHONPATH= uv run mypy src/rallylens/
```

E2E smoke test baseline (`data/raw/smoke_test.mp4`, synthetic 1280×720@10fps):
`detect-shuttle` → **22 raw → 22 ensemble → 28 rectified**; `detect-events` →
**5 events** (3 hit, 1 landing, 1 out_of_frame); `detect-shots` → **3 shots
(clear)**. If ensemble point count drops below non-ensemble (`--no-ensemble`
comparison), the ensemble change is wrong — revert to vote clustering.

## Attribution summary

| Component | Source | License |
|---|---|---|
| TrackNetV3 + weights | qaz812345/TrackNetV3 | MIT © 2024 |
| Event rule bank | ZSHYC/BadmintonTrackNet | MIT © 2024 |
| Pipeline structure | YeonSeong-Lee/rallylens | MIT (declared) |
| Ensemble/floor-gate/shot ideas | amramer/Badminton-visionAI | **proprietary — concepts only, NO code copied** |
| YOLO11 weights, `web/` | ultralytics | AGPL-3.0 |