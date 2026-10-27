# RallyLens browser model assets

This directory is intentionally empty in source control. The repository's current
TrackNet checkpoint has not been cleared for browser redistribution and must not be
copied into a Pages build.

A release manifest may list only models that:

1. pass the public golden-set thresholds documented in `web/README.md`;
2. have a verified redistribution license and source URL;
3. use the fixed browser shapes (`[1,3,640,640]` or `[1,3,960,960]` for YOLO;
   `[1,27,288,512]` to `[1,8,288,512]` for TrackNet); and
4. match the SHA-256 in this manifest.

The Pages release job is expected to download approved ONNX assets from a GitHub
Release, verify them, and rewrite their URLs to local files. An empty manifest makes
the app fail closed with `MODEL_UNAVAILABLE`; it never substitutes fake analysis.
