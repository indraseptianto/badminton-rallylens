# Live coaching mode — plan and gates

Goal: a coach gets an evidence-based briefing during a game interval (~60 s), not
only a post-match report. v1 targets a **fixed camera at the venue**; broadcast
feeds (replays, cuts, pan/zoom) are a separate, harder problem and out of scope.

## Why a benchmark comes first

Live mode only works if the pipeline sustains more frames per second than the
camera produces. Everything else (streaming, briefing, UI) is wasted if it cannot.
`rallylens bench` measures this on the target machine:

```bash
uv run rallylens bench data/raw/<clip>.mp4 --frames 60
uv run rallylens bench data/raw/<clip>.mp4 --pose-device mps   # or cuda:0
```

It reports decode, pose and TrackNet separately, then two figures:

| Figure | Meaning |
|---|---|
| one worker, stages in turn | `1 / Σ(1/fps_stage)` — the simplest single-process loop |
| one worker per stage | `min(fps_stage)` — optimistic: assumes stages do not share one saturated device |

Each is divided by the video fps to give a real-time factor; the gate defaults to
**1.2** (headroom for spikes).

| Verdict | Meaning | Action |
|---|---|---|
| `PASS` | single worker keeps up | proceed |
| `PASS (parallel only)` | needs a worker per stage | design live mode as a worker pipeline |
| `FAIL` | below real-time | use a faster device, a lighter model, or a frame stride (the report suggests one; accuracy at stride > 1 is **unverified**) |

A disabled tracker (missing TrackNet weights) is refused rather than timed: it
returns instantly and would report meaningless FPS.

Measured so far: TrackNetV3 on a CPU-only cloud sandbox ran at ~1 fps (≈1 s per
frame), i.e. far below real-time. GPU/Apple-Silicon numbers are not yet measured.

## Phases (each gated)

0. **Benchmark** on target hardware. Gate: real-time factor ≥ 1.2.
1. **Replay-as-live harness**: play a recorded video at 1x through a streaming
   pipeline with a bounded queue that drops frames instead of accumulating lag;
   rally/game state machine; incremental metrics. Prerequisite fix:
   `BounceDetector` infers frame size from the maximum observed coordinates
   instead of the real video size, which drifts as data arrives.
2. **Hardening**: view classifier (ignore replays/close-ups), court tracking
   homography, scoreboard OCR or manual score, smoothed speed/acceleration.
3. **Briefing + UI**: three evidence-backed findings and one tactical suggestion
   per interval (change vs. start of game, not raw numbers); report language
   option; coach page over SSE.
4. **Ground truth**: a labelled shot set (~300), confusion matrix, then a trained
   classifier; logging, resume and health checks.

## Honesty rules for metrics shown to a coach

- Every metric carries a confidence; below threshold it is shown as `n/a`.
- Single-camera shuttle speed is an *estimate*: published monocular smash-speed
  work reports a mean absolute error of ~66 km/h against a radar gun (20 smashes,
  one player). Show it as relative, never as a measured km/h.
- Player speed/acceleration must be smoothed before display; raw position
  differentiation produces non-physical accelerations.
- Players swap sides between games; identity must follow a game state machine,
  not "upper/lower" labels.

## Phase 1 harness (implemented): replay-as-live

`rallylens live <video> --from-track [--brief-every 60] [--speed 1]` replays a saved
shuttle track frame by frame through the same components a camera loop will use:

- `live/incremental.py` — re-runs the batch `BounceDetector` and releases an event only
  after `settle_s` (0.3 s) of newer frames exist, so live and batch results match
  (tested: `test_incremental_equals_batch`).
- `live/segmenter.py` — events → rallies (first hit → landing, or 4 s of silence).
- `live/briefing.py` — Indonesian coach briefing per interval. Shows `n/a` instead of
  guessing (no calibration → no landing zones; < 5 shots → counts, no percentages).
- `live/queue.py` — drop-oldest bounded queue for the camera loop.

### Camera / stream mode

`rallylens live 0` (webcam), `rallylens live rtsp://… --corners-from <video_id>`, or a file path.
`vision/frame_source.py` opens the source; `live/runner.py` runs a reader thread into a
drop-oldest queue so the models always see fresh frames (dropped count is reported, and
capture indices keep counting so the clock stays correct). Tested with fake sources/models and
a real file; a real camera/RTSP feed and sustained real-time on GPU are untested here.

Pose: `--pose [--pose-device cuda:0]` adds YOLO-pose + ByteTrack per frame, so briefings
attribute shots per tracked player (`n/a` without it). Live/stream sources reconnect with
back-off (5 tries); files do not.

Not done yet: scoreboard input, mapping tracker IDs to named players/sides. Camera mode requires
`rallylens bench` to pass on the target hardware first.
