"""Export the RallyLens 9-frame TrackNet model to its fixed browser ONNX shape."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from rallylens.vision.tracknet import TrackNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = TrackNet(in_dim=27, out_dim=8)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros((1, 27, 288, 512), dtype=torch.float32),
        args.output,
        input_names=["frames"],
        output_names=["heatmaps"],
        opset_version=18,
        dynamo=False,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{digest}  {args.output}")


if __name__ == "__main__":
    main()
