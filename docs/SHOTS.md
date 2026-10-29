# Shot Classification (smash / drive / drop / clear / net)

Local, rule-based stroke classification — the no-cloud-API counterpart of the
Roboflow-hosted shot classifiers used by other badminton projects (e.g.
`amramer/Badminton-visionAI`). It classifies each **hit** event detected by
the event module (`docs/EVENTS.md`) using kinematics computed from the
repaired shuttle track plus player detections. Pure numpy — no model, no
network, no GPU, no API key.

## Why not a cloud API?

Cloud shot classifiers require an API key, send every frame off-box (privacy
& cost), and depend on a third-party model version. The local classifier is
deterministic, offline, and free — at the cost of being a heuristic. It is
the right trade for CPU-only VPS inference and portfolio use.

## Pipeline

```
data/tracks/<id>/<id>_shuttle.jsonl   (from detect-shuttle)
data/events/<id>/<id>_events.jsonl    (from detect-events; hit events only)
data/detections/<id>/<id>_players.jsonl (from detect)
        │
        ▼
classify_shots(events, track, detections, fps, video_size)
        │
        ▼
data/shots/<id>/<id>_shots.jsonl      (list[ShotEvent])
```

`run` runs this stage automatically after events (best-effort).

## ShotEvent schema

```json
{"frame_idx": 39, "player_track_id": null, "shot_type": "clear",
 "confidence": 0.75, "speed_px": 33.2, "vy_after": -3.1, "x": 814, "y": 62}
```

- `speed_px`: max of approach/exit shuttle speed in **pixels/frame**
  (original-video pixels — thresholds below assume ~720p/30fps; tune for
  other resolutions).
- `vy_after`: vertical velocity right after the hit in px/frame
  (positive = falling, negative = rising).
- `player_track_id`: closest player detection within ±15 frames and
  260 px of the hit (None when no player nearby).

## Rule bank (ordered)

| Condition | Shot | Confidence |
|---|---|---|
| `speed ≤ 14` and high contact (`y < 0.35·H`) | **net** | 0.62 |
| `speed ≥ 30` and low contact and `vy_after > 2` | **smash** | 0.85 |
| `speed ≥ 30` and `vy_after > 0` | **smash** | 0.70 |
| `speed ≥ 18` and `\|vy_after\| ≤ 3` | **drive** | 0.75 |
| `speed ≥ 20` and `vy_after < -2` | **clear** | 0.75 |
| `speed ≤ 22` | **drop** | 0.60 |
| otherwise | **unknown** | 0.40 |

Thresholds are the shippable defaults tuned on broadcast-style 720p 30fps
footage. They are module constants (`SMASH_MIN_SPEED`, etc. in
`analysis/shot_classifier.py`) — tune there, not in the pipeline.

## Limitations & tuning

- The classifier reads *speed*, not court position: a smash hit flat at the
  net can be called `clear` if `vy_after` is strongly negative. Court
  homography (`data/calibration/<id>/corners.json`) is not consumed yet —
  wiring it in would let the classifier distinguish net/backcourt shots.
- `player_track_id` is heuristic (IoU-free nearest-center): works for
  singles, ambiguous in doubles.
- `unknown` almost always means the hit event had too little track context —
  raise `min_visible_before` in the event detector first before lowering
  thresholds here.
- Add `docs/TRACKNETV3.md`-style experiments here if you retune for other
  footage, so the mapping between threshold changes and observed outcomes is
  recorded.