# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RallyLens — a CLI pipeline that turns a badminton match video (local file or YouTube URL) into player tracks, shuttlecock tracks, court calibration, and visualizations. Built on YOLO11-pose, ByteTrack, and TrackNetV3. Python 3.11+, managed with `uv`.

## Commands

```bash
uv sync                         # install deps (after first clone; needs ffmpeg on PATH)
uv run rallylens <subcommand>   # run CLI (entry point: rallylens.cli:main)

uv run pytest                            # full test suite
uv run pytest tests/test_common.py       # single file
uv run pytest tests/test_common.py::test_name -q   # single test
uv run ruff check src/                   # lint
uv run ruff format src/                  # format
uv run mypy                              # strict type-check (config in pyproject)
```

CLI subcommands (each writes to a fixed location under `data/`):

- `ingest <youtube-url>` — download to `data/raw/<video_id>.mp4`. Supports `--start/--end` to clip.
- `detect <video>` — YOLO11-pose + ByteTrack → `data/detections/<stem>/<stem>_players.jsonl`.
- `detect-shuttle <video>` — TrackNetV3 → `data/tracks/<stem>/<stem>_shuttle.jsonl`.
- `detect-events <video>` — landing / hit / out_of_frame events from the shuttle track → `data/events/<stem>/`. `--fps` defaults to the video's own.
- `detect-shots <video>` — rule-based stroke labels (smash/drive/drop/clear/net) for hit events → `data/shots/<stem>/`.
- `report <video>` — LLM rally report (needs cloud credentials).
- `bench <video>` — throughput gate: can this machine run pose + TrackNet at >= 1.2x real time? Exits 1 on fail.
- `live <camera-index|url|file>` — coach briefings every `--brief-every` s; `--pose` adds player attribution, `--from-track` replays saved tracks without models. See `docs/LIVE.md`.
- `calibrate <video>` — Hough-based court corners → `data/calibration/<stem>/corners.json`. `--interactive` opens an OpenCV picker.
- `viz <video>` — needs all three above; renders overlay MP4 and court diagram GIF (with heatmap background) into `data/viz/<stem>/`.
- `run <url-or-path>` — one-shot ingest + detect.

`RALLYLENS_LOG_LEVEL` env var controls log verbosity (default `INFO`).

## Architecture

The codebase is layered so that each concern lives in exactly one place. When extending, respect the boundaries below — most bugs come from leaking I/O, paths, or framework objects across them.

**`rallylens.config`** — the *only* module that knows on-disk paths. Pure constants, zero imports from the rest of the package. Anything that needs a directory imports from here.

**`rallylens.serialization`** — generic pydantic ↔ JSON/JSONL helpers. All pipeline artifacts on disk are pydantic `BaseModel` instances, so persistence is a one-liner. Don't hand-roll JSON elsewhere.

**`rallylens.pipeline.io`** — the single mapping from `video_id` to artifact paths, and typed `save_*` / `load_*` functions for each artifact kind (player detections, shuttle tracks, court corners, viz outputs). `video_id` is the input file stem (e.g. `abc_10s_20s` for a clipped download); each clip owns its own `data/<kind>/<video_id>/` directory. When you add a new artifact type, add its path + load/save here, not in the caller.

**`rallylens.vision`** — the heavy CV layer. Each module owns one model and exports a pydantic result type:
- `detect_track.py` → `Detection` (YOLO11-pose + optional ByteTrack via ultralytics). `select_two_players()` post-processes for singles matches by keeping the 2 most frame-frequent track IDs.
- `tracknet.py` + `shuttle_tracker.py` → `ShuttlePoint` (TrackNetV3, sliding-window inference).
- `court_detector.py` + `court_picker.py` → `CourtCorners` (Hough auto-detect, with an interactive OpenCV fallback).

This is the only layer allowed to import `ultralytics` / `cv2`. mypy is relaxed for `vision.detect_track` because of incomplete upstream stubs — keep new framework-tangled code in that module rather than spreading the relaxation.

**`rallylens.pipeline`** — orchestration. Thin functions that wire `vision` modules to `pipeline.io`. `orchestrator.run_full_pipeline` accepts either a local path or a YouTube URL (it dispatches to `ingest.downloader` only when the path doesn't exist on disk). `shuttle.py` and `court.py` mirror this pattern for their respective stages.

**`rallylens.analysis`** — pure logic on saved artifacts: `events/` (BounceDetector rule bank, rescaled to 512x288 model space), `shot_classifier.py` (speed thresholds tuned at 1280 px / 30 fps and normalized by frame size and fps), `metrics.py`. Must not import `viz`. Court geometry (`compute_homography`, `project_point`) lives in `vision/court_geometry.py`.

**`rallylens.live`** — live-mode building blocks, no models or I/O of their own: `incremental.py` (event detector with a settle delay; equals batch output), `segmenter.py` (events → rallies), `briefing.py` (Indonesian coach briefing; shows `n/a` rather than guessing), `queue.py` (thread-safe drop-oldest), `session.py` (`LiveEngine`, `replay_live`), `runner.py` (reader thread + consumer). Frame sources and models stay in `vision/` (`frame_source.py`, `make_frame_pose`, `ShuttleTracker`).

**`rallylens.viz`** — pure rendering on top of the artifacts produced by `pipeline.io`. Reads detections + shuttle track + court corners; writes overlay MP4 and court diagram GIF (the heatmap is rendered as the GIF's background, not a standalone PNG). No model inference here.

**`rallylens.ingest.downloader`** — yt-dlp wrapper with caching. Clipped downloads are cached under a separate filename (`<video_id>_<start>s_<end>s.mp4`) so full and clipped versions coexist.

**`rallylens.cli`** — click group that's a thin shim over `pipeline.*`. New CLI flags should bottom out in pipeline functions, not in vision modules directly.

**`rallylens.common`** — only cross-cutting helpers (logger, `load_env`, `ensure_dir`, `require_ffmpeg`, `read_video_properties`, `read_frame_at`). Don't grow it; prefer a focused module.

## Conventions

- `tests/test_architecture.py` enforces layer boundaries and the cv2 import allowlist; run it after moving code between packages.
- CI (`.github/workflows/python.yml`) runs ruff check, ruff format --check, mypy and pytest — all four must pass before merge.

- Pydantic v2 everywhere; models are `frozen=True` where used as detection records.
- mypy strict mode is enabled for the whole package except the two `tool.mypy.overrides` blocks in `pyproject.toml`. Don't add new overrides without a strong reason.
- ruff selects `E,F,I,W,B,UP,SIM`; line length 100; `E501` ignored.
- Model weights live in `models/` (gitignored except `.gitkeep`). YOLO is downloaded into `MODELS_DIR` on first run via a `contextlib.chdir` trick to avoid polluting the project root.
- ffmpeg must be on PATH (`brew install ffmpeg` on macOS); `require_ffmpeg()` enforces this at runtime where needed.
- Artifact directory layout is `data/{raw,detections,tracks,calibration,viz}/<video_id>/...` — keep it consistent by going through `pipeline.io`.
