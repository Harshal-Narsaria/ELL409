"""Repeated latency measurements after training, using validation images only."""
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .common import environment, load_checkpoint, seed_everything, write_json
from .data import ImageSplit
from .models import ImageTreeClassifier
from .train import load_tree


def repeated_latency(runs_root, output_dir, device, repetitions=5, iterations=50):
    seed_everything(409)
    result = {"environment": environment(), "device": str(device), "batch_size": 16,
              "repetitions": repetitions, "iterations_per_repetition": iterations,
              "warmup_iterations": 10, "input": "Sixteen validation images spread across all classes; cached and image paths use the same IDs",
              "excludes": "Disk reads and resizing; no model training or test-data access", "datasets": {}}
    def measure(call):
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations): call()
        if device.type == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / iterations / 16
    with torch.inference_mode():
        for dataset in ("cifar10", "cats_vs_dogs"):
            run = Path(runs_root) / dataset
            bundle = load_checkpoint(run / "features.pt")
            data = ImageSplit(Path(runs_root).parent / "data" / dataset / "manifest.json", "validation")
            labels = bundle["splits"]["validation"]["y"]
            indices = [int(i) for label in range(bundle["classes"])
                       for i in (labels == label).nonzero().flatten()[:16 // bundle["classes"] + int(label < 16 % bundle["classes"])]]
            normalized = torch.stack([data[i][0] for i in indices])
            images = (normalized * data.std + data.mean).to(device)
            cached = bundle["splits"]["validation"]["x"][indices].to(device)
            names = ["roots-only", "adaptive-w64", "pruned", "narrow-w32", "narrow-w16"]
            models = {}
            parity = {}
            for name in names:
                tree, _ = load_tree(run / name / "tree.pt", device)
                full = ImageTreeClassifier(tree, bundle).to(device).eval()
                models[name] = (tree.eval(), full)
                # CUDA convolution rounding can differ between cache and inference
                # batch sizes. Record it instead of treating bit parity as guaranteed.
                parity[name] = {"max_score_absolute_difference": float((tree(cached) - full(images)).abs().max()),
                                "prediction_disagreements": int((tree.predict(cached) != full.predict(images)).sum())}
            samples = {name: {"cached": [], "image": []} for name in names}
            for repeat in range(repetitions):
                # Rotate order to reduce systematic warmup/thermal-order bias.
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                for name in order:
                    tree, full = models[name]
                    for _ in range(10): tree.predict(cached); full.predict(images)
                    samples[name]["cached"].append(measure(lambda: tree.predict(cached)))
                    samples[name]["image"].append(measure(lambda: full.predict(images)))
            result["datasets"][dataset] = {}
            for name, kinds in samples.items():
                result["datasets"][dataset][name] = {
                    kind: {"median_ms_per_image": statistics.median(values),
                           "q1_ms_per_image": float(np.quantile(values, .25)),
                           "q3_ms_per_image": float(np.quantile(values, .75)), "samples_ms_per_image": values}
                    for kind, values in kinds.items()}
                result["datasets"][dataset][name]["cache_image_parity"] = parity[name]
            print(f"Repeated inference timings complete: {dataset}", flush=True)
    write_json(Path(output_dir) / "latency-benchmark.json", result)
    return result
