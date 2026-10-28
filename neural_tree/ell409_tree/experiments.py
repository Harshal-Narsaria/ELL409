from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

from .common import digest, environment, load_checkpoint, read_json, save_checkpoint, write_json
from .models import TreeClassifier, ImageTreeClassifier
from .train import (cache_features, cpu_state, feature_metrics, image_metrics, load_tree,
                    train_cnn, train_trees)
from .data import ImageSplit
from torch.utils.data import DataLoader


def inference_benchmark(model, bundle, device, batches=10):
    x = bundle["splits"]["validation"]["x"][:256].to(device)
    with torch.no_grad():
        for _ in range(3):
            model.predict(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(batches):
            model.predict(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    timing = {"cached_feature_ms_per_image": (time.perf_counter() - start) * 1000 / batches / len(x),
              "benchmark_batch_size": len(x), "benchmark_repetitions": batches}
    if "trunk_state" in bundle:
        deployed = ImageTreeClassifier(copy.deepcopy(model).cpu(), bundle).to(device).eval()
        images = torch.zeros(16, 3, bundle["resolution"], bundle["resolution"], device=device)
        with torch.no_grad():
            for _ in range(3): deployed.predict(images)
            if device.type == "cuda": torch.cuda.synchronize()
            begin = time.perf_counter()
            for _ in range(batches): deployed.predict(images)
            if device.type == "cuda": torch.cuda.synchronize()
        timing.update({"image_to_prediction_ms_per_image": (time.perf_counter() - begin) * 1000 / batches / len(images),
                       "image_benchmark_batch_size": len(images), "excludes": "disk read and resize"})
    return timing


def deployment_payload(model, bundle):
    payload = {"spec": model.spec(), "model": cpu_state(model),
               "feature_mean": bundle["feature_mean"], "feature_std": bundle["feature_std"],
               "trunk_hash": bundle["trunk_hash"], "manifest": bundle["manifest"]}
    for key in ("trunk_state", "image_mean", "image_std", "resolution", "class_names"):
        if key in bundle:
            payload[key] = bundle[key]
    return payload


def convergence_diagnostics(tasks):
    diagnostics = []
    for class_key, local in tasks.items():
        for path, task in local.items():
            history = task["base"]["history"]
            if len(history) < 5:
                continue
            tail = history[-min(20, len(history)):]
            relative = (tail[0]["loss"] - tail[-1]["loss"]) / max(abs(tail[0]["loss"]), 1e-8)
            diagnostics.append({"class": class_key, "node": path, "loss_relative_drop": relative,
                                "last_errors": tail[-1]["errors"], "tail_epochs": len(tail),
                                "needs_more_optimization": bool(relative > .01 and tail[-1]["errors"] > 0)})
    return {"nodes": diagnostics, "needs_extension": any(d["needs_more_optimization"] for d in diagnostics),
            "criterion": "Base loss drops >1% over last up to 20 epochs while classification errors remain; diagnostic, not a convergence proof."}


def run_tree(bundle, run, name, config, device):
    folder = Path(run) / name
    output = folder / "metrics.json"
    if output.exists():
        result = read_json(output)
        if result["config"] != config or result["features"] != bundle["fingerprint"]:
            raise ValueError("Experiment identity changed. Choose a new name or run directory.")
        return result
    model = train_trees(bundle, folder, config, device)
    saved = load_checkpoint(folder / "tree.pt")
    seconds = sum(task["base"]["elapsed"] + task.get("correction", {}).get("elapsed", 0)
                  for tasks in saved["tasks"].values() for task in tasks.values())
    result = {"name": name, "kind": "tree", "config": config, "features": bundle["fingerprint"],
              "manifest": bundle["manifest"], "environment": environment(), "training_seconds": seconds,
              "model_bytes": (folder / "tree.pt").stat().st_size,
              "size": model.size(bundle["trunk_parameters"]),
              "train": feature_metrics(model, bundle["splits"]["train"], device),
              "validation": feature_metrics(model, bundle["splits"]["validation"], device),
              "timing": inference_benchmark(model, bundle, device)}
    result["convergence"] = convergence_diagnostics(saved["tasks"])
    # Include a compact deployment bundle distinct from bulky optimizer/task audit checkpoints.
    save_checkpoint(folder / "deployment.pt", deployment_payload(model, bundle))
    result["deployment_bytes"] = (folder / "deployment.pt").stat().st_size
    write_json(output, result)
    print(f"{name}: train {result['train']['accuracy']:.4%}, validation {result['validation']['accuracy']:.4%}", flush=True)
    return result


def prune_tree(bundle, run, source, device, tolerance=.005):
    folder = Path(run) / "pruned"
    if (folder / "metrics.json").exists():
        return read_json(folder / "metrics.json")
    folder.mkdir(parents=True, exist_ok=True)
    model, saved = load_tree(Path(run) / source / "tree.pt", device)
    original_accuracy = feature_metrics(model, bundle["splits"]["validation"], device)["accuracy"]
    floor = original_accuracy - tolerance
    log = []
    # Always compare to the ORIGINAL accuracy: cumulative damage cannot exceed tolerance.
    candidates = [(i, path, depth, side) for i, root in enumerate(model.roots)
                  for path, depth, node in root.walk() for side in ("A", "B")
                  if getattr(node, side) is not None]
    candidates.sort(key=lambda c: -c[2])
    for i, path, depth, side in candidates:
        node = model.roots[i]
        for step in path.split(".")[1:]:
            node = getattr(node, step, None)
            if node is None:
                break
        if node is None or getattr(node, side) is None:
            continue
        child = getattr(node, side)
        setattr(node, side, None)
        accuracy = feature_metrics(model, bundle["splits"]["validation"], device)["accuracy"]
        accepted = accuracy + 1e-12 >= floor
        if not accepted:
            setattr(node, side, child)
        log.append({"class": i, "parent": path, "side": side, "validation_accuracy": accuracy,
                    "accepted": accepted, "accuracy_floor": floor})
    saved.update({"spec": model.spec(), "model": cpu_state(model), "pruning": log, "source": source})
    save_checkpoint(folder / "tree.pt", saved)
    result = {"name": "pruned", "kind": "pruned", "source": source,
              "features": bundle["fingerprint"], "manifest": bundle["manifest"],
              "config": saved["config"], "environment": environment(), "pruning": log,
              "size": model.size(bundle["trunk_parameters"]),
              "train": feature_metrics(model, bundle["splits"]["train"], device),
              "validation": feature_metrics(model, bundle["splits"]["validation"], device),
              "timing": inference_benchmark(model, bundle, device)}
    save_checkpoint(folder / "deployment.pt", deployment_payload(model, bundle))
    result["deployment_bytes"] = (folder / "deployment.pt").stat().st_size
    write_json(folder / "metrics.json", result)
    return result


def sweep(bundle, run, config, device, tag="sweep"):
    records = []
    widths = config.get("sweep_widths", [8, 16, 32, 64, 128, 256, 512])
    for width in widths:
        node_config = {**config, "width": width, "max_depth": 2, "max_nodes": 7, "fixed_topology": True}
        name = f"{tag}-w{width}-s{config['seed']}"
        result = run_tree(bundle, run, name, node_config, device)
        records.append(result)
        write_json(Path(run) / (tag + ".json"), records)
    interpolation = next((i for i, r in enumerate(records) if r["train"]["errors"] == 0), None)
    center = interpolation if interpolation is not None else len(widths) // 2
    selected = sorted(set(max(0, min(len(widths) - 1, i)) for i in (center - 1, center, center + 1)))
    repeats = []
    for i in selected:
        for seed in config.get("repeat_seeds", [config["seed"] + 1, config["seed"] + 2]):
            node_config = {**config, "seed": seed, "width": widths[i], "max_depth": 2,
                           "max_nodes": 7, "fixed_topology": True}
            repeats.append(run_tree(bundle, run, f"{tag}-w{widths[i]}-s{seed}", node_config, device))
    write_json(Path(run) / (tag + "-repeats.json"), repeats)
    return records, repeats


def plot_results(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    for path in run.glob("sweep*.json"):
        if "repeats" in path.name or "interpretation" in path.name:
            continue
        records = read_json(path)
        if not records:
            continue
        repeat_path = path.with_name(path.stem + "-repeats.json")
        all_records = records + (read_json(repeat_path) if repeat_path.exists() else [])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for split, label in [("train", "Train"), ("validation", "Validation")]:
            xs, means, deviations, losses, loss_sd = [], [], [], [], []
            for record in records:
                same = [r for r in all_records if r["config"]["width"] == record["config"]["width"]]
                import numpy as np
                errors = [1 - r[split]["accuracy"] for r in same]
                values = [r[split]["loss"] for r in same]
                xs.append(record["size"]["allocated_parameters"])
                means.append(np.mean(errors)); deviations.append(np.std(errors))
                losses.append(np.mean(values)); loss_sd.append(np.std(values))
            axes[0].errorbar(xs, means, yerr=deviations, marker="o", label=label)
            axes[1].errorbar(xs, losses, yerr=loss_sd, marker="o", label=label)
        first = next((r for r in records if r["train"]["errors"] == 0), None)
        for ax in axes:
            ax.set_xscale("log"); ax.set_xlabel("Allocated parameters, shared trunk counted once")
            ax.grid(alpha=.2); ax.legend()
            if first:
                ax.axvline(first["size"]["allocated_parameters"], color="gray", linestyle="--")
        axes[0].set_ylabel("Classification error"); axes[1].set_ylabel("Cross-entropy")
        fig.suptitle(f"{path.stem}: observed capacity sweep; bars = seed standard deviation")
        fig.tight_layout(); fig.savefig(figures / (path.stem + ".png"), dpi=160); plt.close(fig)
        summary = interpret_sweep(records, all_records)
        write_json(run / (path.stem + "-interpretation.json"), summary)
    history_path = run / "cnn-history.json"
    if history_path.exists():
        history = read_json(history_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        for split in ("train", "validation"):
            ax.plot([r["epoch"] for r in history], [1 - r[split]["accuracy"] for r in history], label=split)
        ax.set(xlabel="CNN training epoch", ylabel="Classification error", title="Epoch-wise CNN error (not the tree)")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(figures / "cnn-epochs.png", dpi=160); plt.close(fig)


def interpret_sweep(primary, all_records):
    """Qualitative evidence summary; never manufacture the desired phenomenon."""
    import numpy as np
    index = next((i for i, r in enumerate(primary) if r["train"]["errors"] == 0), None)
    means, stderr, counts = [], [], []
    for record in primary:
        values = [1 - r["validation"]["accuracy"] for r in all_records if r["config"]["width"] == record["config"]["width"]]
        means.append(float(np.mean(values)))
        stderr.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.)
        counts.append(len(values))
    summary = {"widths": [r["config"]["width"] for r in primary], "validation_error_mean": means,
               "validation_error_standard_error": stderr, "seed_counts": counts,
               "primary_seed_interpolation_index": index, "candidate_double_descent": False,
               "caution": "A qualitative pattern check, not a statistical test or convergence proof."}
    if index is None:
        summary["observation"] = "No primary configuration reached zero training error; the post-interpolation regime was not established."
    elif index == 0:
        summary["observation"] = "The smallest primary model already interpolates; this sweep does not show the initial underparameterized regime."
    elif index == len(primary) - 1:
        summary["observation"] = "Interpolation occurs only at the largest tested primary width; additional post-threshold capacity is needed."
    else:
        peaks = range(max(2, index - 1), min(len(primary) - 1, index + 2))
        for peak in peaks:
            earlier = min(range(1, peak), key=lambda i: means[i])
            later = min(range(peak + 1, len(primary)), key=lambda i: means[i])
            tolerance = max(.005, 2 * max(stderr[peak], stderr[earlier], stderr[later]))
            if means[0] - means[earlier] > tolerance and means[peak] - means[earlier] > tolerance and means[peak] - means[later] > tolerance:
                summary.update({"candidate_double_descent": True, "peak_index": peak, "earlier_minimum_index": earlier,
                                "later_minimum_index": later, "pattern_tolerance": tolerance})
                break
        summary["observation"] = ("A qualitative first descent, near-threshold rise, and later descent is present; assess optimization flags and seed variability before claiming double descent."
                                   if summary["candidate_double_descent"] else "No clear double-descent pattern was found under this predeclared qualitative criterion.")
    return summary


def finalize_tests(manifest_path, run, config, device):
    """Lock selected candidates before any final held-out evaluation; reuse existing outputs."""
    run = Path(run)
    if not (run / "selected-tree.json").exists() or not (run / "sweep.json").exists() or not (run / "pruned" / "metrics.json").exists():
        raise ValueError("Finish tree, sweep, and compression stages before final test evaluation")
    candidates = sorted(p.parent.name for p in run.glob("*/metrics.json"))
    declaration = {"manifest": read_json(manifest_path)["fingerprint"], "candidates": candidates,
                   "configuration": digest(config)}
    lock = run / "test-declaration.json"
    if lock.exists() and read_json(lock) != declaration:
        raise ValueError("Final evaluation was declared already; do not add or tune candidates using test results")
    write_json(lock, declaration)
    bundles = {name: load_checkpoint(run / name) for name in ("features.pt", "features-early.pt") if (run / name).exists()}
    for candidate in candidates:
        path = run / candidate / "test-metrics.json"
        if path.exists():
            continue
        model, checkpoint = load_tree(run / candidate / "tree.pt", device)
        bundle = next((b for b in bundles.values() if b["fingerprint"] == checkpoint["features"]), None)
        if bundle is None:
            raise ValueError("Matching frozen feature bundle is missing")
        write_json(path, {"declaration": digest(declaration), **feature_metrics(model, bundle["splits"]["test"], device)})
    if not (run / "cnn-test.json").exists():
        from .models import SharedCNN
        manifest = read_json(manifest_path)
        cnn = SharedCNN(len(manifest["classes"])).to(device)
        cnn.load_state_dict(load_checkpoint(run / "cnn-best.pt")["model"])
        loader = DataLoader(ImageSplit(manifest_path, "test"), batch_size=config["image_batch_size"])
        write_json(run / "cnn-test.json", image_metrics(cnn, loader, device))
    write_json(run / "study-complete.json", {"status": "experiments_complete", "target": read_json(run / "selected-tree.json")})


def pipeline(manifest_path, run, config, device, stage="all", finalize=False):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    record = {"config": config, "manifest": read_json(manifest_path)["fingerprint"]}
    if (run / "configuration.json").exists() and read_json(run / "configuration.json") != record:
        raise ValueError("Run directory belongs to another configuration or dataset")
    write_json(run / "configuration.json", record)
    manifest = read_json(manifest_path)
    write_json(run / "data-audit.json", {key: value for key, value in manifest.items() if key != "splits"})
    write_json(run / "environment.json", environment())
    if (run / "test-declaration.json").exists() and stage != "plots":
        if not (run / "study-complete.json").exists():
            raise ValueError("Test candidates are locked. Resume only final test evaluation via the finalize command.")
        return
    if stage in ("all", "cnn"):
        model = train_cnn(manifest_path, run, config, device)
        kwargs = {"batch_size": config["image_batch_size"], "num_workers": config.get("workers", 0)}
        baseline = {split: image_metrics(model, DataLoader(ImageSplit(manifest_path, split), **kwargs), device)
                    for split in ("train", "validation")}
        baseline["parameters"] = sum(p.numel() for p in model.parameters())
        write_json(run / "cnn-baseline.json", baseline)
    if stage in ("all", "features"):
        cache_features(manifest_path, run / "cnn-best.pt", run / "features.pt", config["image_batch_size"], device)
        if (run / "cnn-early.pt").exists():
            cache_features(manifest_path, run / "cnn-early.pt", run / "features-early.pt", config["image_batch_size"], device)
    bundle = None
    if stage in ("all", "trees", "sweep", "compress"):
        bundle = load_checkpoint(run / "features.pt")
    if stage in ("all", "trees"):
        run_tree(bundle, run, "roots-only", {**config, "max_depth": 0, "max_nodes": 1}, device)
        result = run_tree(bundle, run, f"adaptive-w{config['width']}", config, device)
        selected = result["name"]
        for width, depth, nodes in [(128, 4, 31), (256, 4, 31), (256, 5, 63)]:
            if result["train"]["accuracy"] >= config["near_perfect"]:
                break
            new = run_tree(bundle, run, f"adaptive-w{width}-d{depth}",
                           {**config, "width": width, "max_depth": depth, "max_nodes": nodes}, device)
            if (new["train"]["errors"], new["train"]["loss"]) < (result["train"]["errors"], result["train"]["loss"]):
                result, selected = new, new["name"]
        write_json(run / "selected-tree.json", {"name": selected, "target_met": result["train"]["accuracy"] >= config["near_perfect"]})
    if stage in ("all", "sweep"):
        records, _ = sweep(bundle, run, config, device)
        if any(r.get("convergence", {}).get("needs_extension", False) for r in records) and config.get("max_sweep_epochs", 300) > config["node_epochs"]:
            longer = {**config, "node_epochs": config.get("max_sweep_epochs", 300)}
            sweep(bundle, run, longer, device, "sweep-long")
        if records[0]["train"]["errors"] == 0 and (run / "features-early.pt").exists():
            sweep(load_checkpoint(run / "features-early.pt"), run, config, device, "sweep-early")
    if stage in ("all", "compress"):
        selected = read_json(run / "selected-tree.json")["name"]
        source_config = read_json(run / selected / "metrics.json")["config"]
        prune_tree(bundle, run, selected, device)
        for width in (32, 16):
            run_tree(bundle, run, f"narrow-w{width}", {**source_config, "width": width}, device)
    plot_results(run)
    if stage == "all":
        from .visualize import visualize_nodes
        visualize_nodes(manifest_path, run, device)
        if finalize:
            finalize_tests(manifest_path, run, config, device)
