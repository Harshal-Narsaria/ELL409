from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Independent pixel / selective-CNN correction-tree study")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--original", default="../neural_tree")
        p.add_argument("--study-dir", default="runs/smoke-verified" if name == "smoke" else "runs/image-tree-v2")
        p.add_argument("--output-dir", default="results/smoke-verified" if name == "smoke" else "results/image-tree-v2")
        p.add_argument("--config", default="configs/study.json")
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        p.add_argument("--stage", default="main", choices=["main", "sweep", "all"])
    p = sub.add_parser("predict")
    p.add_argument("--deployment", required=True); p.add_argument("--image", required=True)
    args = parser.parse_args()
    if args.command in ("run", "smoke"):
        from .study import study_main
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        study_main(args.original, args.study_dir, args.output_dir, args.config, device, args.stage, args.command == "smoke")
    elif args.command == "predict":
        import numpy as np
        from PIL import Image
        from .common import load_checkpoint
        from .models import ImageForest
        saved = load_checkpoint(args.deployment)
        model = ImageForest.from_spec(saved["spec"])
        model.load_state_dict(saved["model"]); model.eval()
        with Image.open(args.image) as im:
            im = im.convert("RGB").resize((saved["resolution"], saved["resolution"]), Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1)[None] / 255
        image = (image - saved["mean"][None, :, None, None]) / saved["std"][None, :, None, None]
        with torch.no_grad():
            label = int(model.predict(image)[0])
        print(json.dumps({"class": saved["classes"][label], "index": label}))


if __name__ == "__main__":
    main()
