from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import digest, file_hash, load_checkpoint, read_json, save_checkpoint, tensor_hash


def make_cache(manifest_path, destination, resolution=32, limit=None):
    """Cache pixels, never a learned representation; no test image is opened."""
    manifest_path, destination = Path(manifest_path), Path(destination)
    manifest = read_json(manifest_path)
    signature = digest({"manifest_file": file_hash(manifest_path), "resolution": resolution, "limit": limit})
    if destination.exists():
        cache = load_checkpoint(destination)
        if cache["signature"] != signature:
            raise ValueError("Pixel cache identity changed; choose a new study directory.")
        return cache
    folder = manifest_path.parent
    arrays = np.load(folder / "images.npy", mmap_mode="r") if (folder / "images.npy").exists() else None
    cache = {"signature": signature, "manifest": manifest["fingerprint"], "classes": manifest["classes"], "source": manifest["source"],
             "resolution": resolution, "splits": {}, "limit": limit, "uses_test_data": False}
    for split in ("train", "validation"):
        records = manifest["splits"][split]
        if limit is not None:
            # Saved manifests are grouped by class; a prefix can contain only one class.
            by_class = {label: [r for r in records if r["label"] == label] for label in range(len(manifest["classes"]))}
            selected, position = [], 0
            while len(selected) < min(limit, len(records)):
                for label in by_class:
                    if position < len(by_class[label]) and len(selected) < limit:
                        selected.append(by_class[label][position])
                position += 1
            records = selected
        pixels = torch.empty((len(records), 3, resolution, resolution), dtype=torch.uint8)
        labels = torch.tensor([r["label"] for r in records], dtype=torch.long)
        for i, record in enumerate(records):
            if "index" in record:
                image = Image.fromarray(arrays[record["index"]])
            else:
                with Image.open(folder / record["path"]) as im:
                    image = im.convert("RGB")
            image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
            pixels[i] = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        cache["splits"][split] = {"pixels": pixels, "y": labels, "ids": [r["id"] for r in records]}
    sums, squares, count = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0
    for batch in cache["splits"]["train"]["pixels"].split(256):
        x = batch.to(torch.float64) / 255
        sums += x.sum((0, 2, 3))
        squares += x.square().sum((0, 2, 3))
        count += len(x) * resolution**2
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    cache["mean"], cache["std"] = mean.float(), std.float()
    import hashlib
    from collections import Counter, defaultdict
    groups = defaultdict(Counter)
    for pixels, label in zip(cache["splits"]["train"]["pixels"], cache["splits"]["train"]["y"].tolist()):
        groups[hashlib.sha256(pixels.numpy().tobytes()).digest()][label] += 1
    unavoidable = sum(sum(labels.values()) - max(labels.values()) for labels in groups.values())
    cache["pixel_diagnostics"] = {"unique_training_inputs": len(groups),
        "conflicting_input_groups": sum(len(labels) > 1 for labels in groups.values()),
        "minimum_errors_from_exact_pixel_collisions": unavoidable,
        "training_accuracy_upper_bound": 1 - unavoidable / len(cache["splits"]["train"]["y"])}
    cache["fingerprint"] = digest({"signature": signature, "train": tensor_hash(cache["splits"]["train"]["pixels"], cache["splits"]["train"]["y"]),
                                   "validation": tensor_hash(cache["splits"]["validation"]["pixels"], cache["splits"]["validation"]["y"])})
    save_checkpoint(destination, cache)
    return cache


def normalized(pixels, cache, device):
    x = pixels.to(device=device, dtype=torch.float32) / 255
    return (x - cache["mean"].to(device)[None, :, None, None]) / cache["std"].to(device)[None, :, None, None]
