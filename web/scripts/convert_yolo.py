"""Export one of the fixed RallyLens YOLO pose profiles using Ultralytics."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROFILES = {"yolo11n-pose-640": ("yolo11n-pose.pt", 640), "yolo11s-pose-960": ("yolo11s-pose.pt", 960)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    weights, size = PROFILES[args.profile]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(YOLO(weights).export(format="onnx", imgsz=size, batch=1, dynamic=False, simplify=True, opset=18))
    target = args.output_dir / f"{args.profile}.onnx"
    exported.replace(target)
    print(target)


if __name__ == "__main__":
    main()
