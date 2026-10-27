# RallyLens Web

RallyLens Web is a static React application for private, on-device badminton
analysis. Video bytes and derived tracks remain in memory and are never sent to a
RallyLens server. Vision inference runs in a dedicated worker with ONNX Runtime Web
and WebGPU. The optional English report uses Chrome's built-in LanguageModel API.

## Local development

```bash
cd web
npm ci
npm test
npm run build
npm run dev
```

The checked-in model manifest is deliberately empty because the existing TrackNet
checkpoint has not been cleared for public redistribution. This means the UI and
input/calibration workflow run, but analysis fails closed with actionable guidance.
See `public/models/README.md` and `scripts/` for the release process.

## Supported profile

- Desktop Google Chrome 148 or newer with WebGPU and WebCodecs
- H.264 MP4 or VP9 WebM, up to 1920×1080 and 60 fps
- Fixed-camera singles or doubles clips of at most 60 seconds
- Local files or CORS-enabled HTTPS URLs supporting byte ranges

No mobile browsers, Safari, Firefox, platform page URLs, moving cameras, 4K,
accounts, backend storage, or WASM inference fallback are supported.

## Release gates

A model-bearing build is releasable only after a redistributable public set of at
least 12 singles and 12 doubles videos demonstrates: player recall ≥95%, IDF1 ≥90%,
shuttle recall ≥85%, court success ≥90%, and core metric error ≤5%. Court and shuttle
tolerance is 1% of the 1080p image diagonal. The TrackNet 8-frame stride must pass
the same comparison. Browser/Python golden tensors cover preprocessing, outputs,
NMS, coordinates, homography, shot events, and metrics.

Real-device release testing must finish analysis plus rendering of a 60-second 1080p
clip within 180 seconds. The startup benchmark rejects devices predicted to miss
that limit.

## Privacy and persistence

Only versioned model files are placed in Cache Storage. Videos and results are not
stored in IndexedDB or uploaded, and sessions cannot be recovered after refresh.
Anonymous diagnostics omit the local filename and source URL.

This web application is licensed under AGPL-3.0-only. Model weights retain their own
licenses and may be published only after explicit redistribution review.
