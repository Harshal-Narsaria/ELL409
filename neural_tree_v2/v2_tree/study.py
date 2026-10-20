from __future__ import annotations

import copy
import csv
import gc
import json
import os
import platform
import shutil
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from .common import (cpu_state, digest, file_hash, load_checkpoint, read_json,
                     save_checkpoint, study_lock, write_json)
from .data import make_cache, normalized
from .models import ImageForest
from .train import evaluate, root_ablation, train_forest


def protected_originals(original):
    original = Path(original).resolve()
    paths = []
    for directory in ("results", "output"):
        paths += [p for p in (original / directory).rglob("*") if p.is_file()]
    paths += list((original / "runs").rglob("*.json"))
    paths += list((original / "dist").glob("*.zip"))
    paths += list((original / "github-upload").glob("*/UPLOAD_MANIFEST.json"))
    paths += list((original / "data").glob("*/manifest.json"))
    return {str(p): file_hash(p) for p in sorted(set(paths))}


def verify_originals(snapshot):
    for path, expected in snapshot.items():
        if not Path(path).exists() or file_hash(path) != expected:
            raise ValueError("A protected original result changed: " + path)


def fixed_definition(path, definition):
    if Path(path).exists():
        if read_json(path) != definition:
            raise ValueError("Frozen study definition changed; choose a separate study directory.")
    else:
        write_json(path, definition)


def load_model(folder, device):
    saved = load_checkpoint(Path(folder) / "deployment.pt")
    model = ImageForest.from_spec(saved["spec"]).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


def convergence(tasks):
    flags = []
    for c, local in tasks.items():
        for path, t in local.items():
            tail = t["base"]["history"][-20:]
            if len(tail) >= 5 and tail[-1]["errors"] > 0:
                drop = (tail[0]["loss"] - tail[-1]["loss"]) / max(tail[0]["loss"], 1e-8)
                if drop > .01:
                    flags.append({"class": c, "node": path, "relative_loss_drop": drop, "last_errors": tail[-1]["errors"]})
    return {"nodes_needing_more_optimization": flags, "convergence_proven": False}


def measured(model, x, y, vx, vy, cache, config, name):
    tr, va = evaluate(model, x, y, config["eval_batch"]), evaluate(model, vx, vy, config["eval_batch"])
    batch = vx[:min(64, len(vx))]
    with torch.no_grad():
        for _ in range(2):
            model(batch)
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            model(batch)
        if x.is_cuda:
            torch.cuda.synchronize()
    return {"name": name, "config": config, "pixels": cache["fingerprint"], "manifest": cache["manifest"],
            "train": tr, "validation": va, "size": model.size(), "uses_test_data": False,
            "inference_ms_per_image": (time.perf_counter() - start) * 1000 / (5 * len(batch)),
            "timing_excludes": "disk loading, image resize and normalization"}


def run_model(folder, cache, x, y, vx, vy, config, name, progress):
    folder = Path(folder)
    result_path = folder / "metrics.json"
    if result_path.exists():
        old = read_json(result_path)
        if old["config"] != config or old["pixels"] != cache["fingerprint"]:
            raise ValueError("Completed model identity changed")
        return old
    model, tasks = train_forest(x, y, folder, config, cache["fingerprint"], progress)
    record = measured(model, x, y, vx, vy, cache, config, name)
    roots = root_ablation(model)
    record["root_ablation"] = {"train": evaluate(roots, x, y, config["eval_batch"]),
                               "validation": evaluate(roots, vx, vy, config["eval_batch"]), "size": roots.size(),
                               "definition": "Identical learned root weights; children, signed corrections and offsets removed."}
    record["tree_gain_training_pp"] = 100 * (record["train"]["accuracy"] - record["root_ablation"]["train"]["accuracy"])
    record["tree_gain_validation_pp"] = 100 * (record["validation"]["accuracy"] - record["root_ablation"]["validation"]["accuracy"])
    record["convergence"] = convergence(tasks)
    record["training_seconds"] = sum(t["base"]["elapsed"] + t.get("correction", {}).get("elapsed", 0)
                                     for local in tasks.values() for t in local.values())
    save_checkpoint(folder / "deployment.pt", {"spec": model.spec(), "model": cpu_state(model),
                    "mean": cache["mean"], "std": cache["std"], "resolution": cache["resolution"],
                    "classes": cache["classes"], "manifest": cache["manifest"], "pixels": cache["fingerprint"]})
    save_checkpoint(folder / "audit.pt", {"tasks": tasks})
    record["model_bytes"] = (folder / "deployment.pt").stat().st_size
    write_json(result_path, record)
    write_json(folder / "node-histories.json", {c: {p: {"kind": t["kind"], "examples": len(t["labels"]),
                  "positive_examples": int(t["labels"].sum()), "errors_before": t["errors_before"],
                  "errors_after": t["errors_after"], "base": t["base"], "correction": t.get("correction")}
                  for p, t in local.items()} for c, local in tasks.items()})
    print(f"FINISHED {name}: train {record['train']['accuracy']:.3%}, validation {record['validation']['accuracy']:.3%}; tree gains {record['tree_gain_training_pp']:.3f}/{record['tree_gain_validation_pp']:.3f} pp", flush=True)
    return record


def prune(folder, source, cache, x, y, vx, vy, config):
    folder = Path(folder)
    if (folder / "metrics.json").exists():
        return read_json(folder / "metrics.json")
    model, saved = load_model(source, x.device)
    original = evaluate(model, vx, vy, config["eval_batch"])["accuracy"]
    floor = original - config["pruning_tolerance"]
    candidates = [(c, path, depth, side) for c, root in enumerate(model.roots) for path, depth, node in root.walk()
                  for side in ("A", "B") if getattr(node, side) is not None]
    candidates.sort(key=lambda row: -row[2])
    log = []
    for c, path, _, side in candidates:
        node = model.roots[c]
        for step in path.split(".")[1:]:
            node = getattr(node, step, None)
            if node is None:
                break
        if node is None or getattr(node, side) is None:
            continue
        child = getattr(node, side)
        setattr(node, side, None)
        accuracy = evaluate(model, vx, vy, config["eval_batch"])["accuracy"]
        training_accuracy = evaluate(model, x, y, config["eval_batch"])["accuracy"] if config.get("preserve_training_fit_when_pruning") else None
        accepted = accuracy + 1e-12 >= floor and (training_accuracy is None or training_accuracy >= config["near_perfect"])
        if not accepted:
            setattr(node, side, child)
        log.append({"class": c, "node": path, "side": side, "accepted": accepted, "validation_accuracy": accuracy,
                    "training_accuracy": training_accuracy})
    record = measured(model, x, y, vx, vy, cache, config, folder.name)
    record.update({"source": str(source), "pruning": log, "original_validation_accuracy": original,
                   "minimum_allowed_validation_accuracy": floor})
    saved.update({"spec": model.spec(), "model": cpu_state(model)})
    save_checkpoint(folder / "deployment.pt", saved)
    record["model_bytes"] = (folder / "deployment.pt").stat().st_size
    write_json(folder / "metrics.json", record)
    return record


def export_results(study, output, configuration):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for ds in configuration["datasets"]:
        for path in sorted((Path(study) / ds).glob("*/metrics.json")):
            record = read_json(path)
            record["dataset"] = ds
            records.append(record)
    write_json(output / "measurements.json", records)
    fields = ["dataset", "name", "architecture", "width", "root_width", "parameters", "nodes", "cnn_nodes", "max_depth",
              "train_accuracy_percent", "validation_accuracy_percent", "train_loss", "validation_loss",
              "tree_gain_training_pp", "tree_gain_validation_pp", "training_seconds", "model_bytes"]
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({"dataset": r["dataset"], "name": r["name"], "architecture": r["config"]["architecture"],
                 "width": r["config"]["width"], "root_width": r["config"].get("root_width", r["config"]["width"]) if r["config"]["architecture"] != "cnn_root" else r["config"]["width"],
                 **{k: r["size"][k] for k in ("parameters", "nodes", "cnn_nodes", "max_depth")},
                 "train_accuracy_percent": 100 * r["train"]["accuracy"], "validation_accuracy_percent": 100 * r["validation"]["accuracy"],
                 "train_loss": r["train"]["loss"], "validation_loss": r["validation"]["loss"],
                 **{k: r.get(k) for k in ("tree_gain_training_pp", "tree_gain_validation_pp", "training_seconds", "model_bytes")}})
    from .visualize import plot_measurements
    plot_measurements(records, output)
    lines = ["# Version 2 results", "", configuration["policy"], "",
             "All accuracies below use the original labels. Test evaluation is pending an explicit candidate lock.", "",
             "| Dataset | Model | Train % | Validation % | Parameters | Nodes / CNN nodes | Root-to-tree train gain (pp) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    if "root_width" in configuration:
        lines[4:4] = [f"Tree roots have a fixed width of {configuration['root_width']} and {configuration['root_epochs']} epochs. Reported width varies correction-node capacity. CNN-only and parameter-matched controls are unrestricted.", ""]
    for r in records:
        lines.append(f"| {r['dataset']} | {r['name']} | {100*r['train']['accuracy']:.3f} | {100*r['validation']['accuracy']:.3f} | {r['size']['parameters']} | {r['size']['nodes']} / {r['size']['cnn_nodes']} | {r.get('tree_gain_training_pp', 0):.3f} |")
    lines += ["", "The root ablation uses identical root weights and removes correction children. Parameter-matched MLP controls distinguish tree structure from additional capacity. CNN-only roots use the same small CNN family and node budget, not the earlier shared CNN.", "",
              "No gain or double-descent result is assumed. Adaptive child growth changes topology across widths; nominal width is the controlled factor, and actual model size is also recorded. One seed provides exploratory evidence only. Ongoing loss reduction is flagged; a fixed epoch budget is not a convergence proof.", "",
              "Both datasets use 32x32 inputs. The old Cats-vs-Dogs experiment used 128x128; compare Version 2 architectures within Version 2, and disclose this resolution difference when comparing versions.", "",
              "Pruning uses a cumulative validation tolerance of 0.5 percentage points relative to the original model. Training accuracy after pruning is recorded even if interpolation is lost."]
    if configuration.get("report_fit_preserving_prune"):
        lines += ["", "The pruned-retain-fit condition additionally rejects removals that reduce training accuracy below 99.5%, when a near-fitting source was found."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def study_main(original, study, output, config_path, device, stage="main", smoke=False):
    original, study, output = Path(original).resolve(), Path(study).resolve(), Path(output).resolve()
    if original == study or original in study.parents or original == output or original in output.parents:
        raise ValueError("Version 2 outputs must be outside the original project.")
    configuration = read_json(config_path)
    if configuration.get("label_noise", 0) != 0:
        raise ValueError("This version is the original-label experiment; noise needs a separate declared study.")
    if stage not in ("main", "sweep", "all"):
        raise ValueError("Invalid stage")
    if smoke:
        configuration = {**configuration, "datasets": ["synthetic"], "resolution": 16, "width": 4, "max_depth": 1,
                         "max_nodes": 3, "mlp_epochs": 3, "cnn_epochs": 3, "correction_epochs": 5,
                         "escalation": None, "sweep_widths": [2, 4], "sweep_depth": 1, "sweep_nodes": 3}
    status = {"status": "preparing", "pid": os.getpid(), "updated_unix": time.time(), "stage": stage,
              "smoke": smoke, "uses_test_data": False}
    study.mkdir(parents=True, exist_ok=True)
    with study_lock(study / "runner.lock"):
        snapshot_path = study / "preserved-originals.json"
        if snapshot_path.exists():
            snapshot = read_json(snapshot_path)
            verify_originals(snapshot)
        else:
            snapshot = protected_originals(original)
            write_json(snapshot_path, snapshot)
        code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        fixed_definition(study / "definition.json", {"configuration": configuration, "code": code,
                         "original": str(original), "smoke": smoke,
                         "manifests": {ds: file_hash(original / "data" / ds / "manifest.json") for ds in configuration["datasets"]}})
        write_json(study / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                    "device": str(device), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
        last_progress_write = [0.]
        def progress(**fields):
            node_changed = "node" in fields and fields["node"] != status.get("node")
            status.update(fields)
            status["updated_unix"] = time.time()
            terminal_epoch = fields.get("epoch") == fields.get("epochs") and "epoch" in fields
            if "epoch" in fields and not node_changed and not terminal_epoch and time.time() - last_progress_write[0] < 5:
                return
            try:
                write_json(study / "progress.json", status)
                last_progress_write[0] = time.time()
            except PermissionError as exc:
                # Progress is observational: a locked status file must not lose a training run.
                print(f"Status file remains temporarily locked; training/checkpoints continue: {exc}", flush=True)
        try:
            for ds in configuration["datasets"]:
                progress(status="preparing_pixels", dataset=ds)
                if configuration.get("reuse_pixel_cache_from"):
                    source_pixels = Path(configuration["reuse_pixel_cache_from"]) / ds / "pixels.pt"
                    destination_pixels = study / ds / "pixels.pt"
                    if source_pixels.exists() and not destination_pixels.exists():
                        destination_pixels.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_pixels, destination_pixels)
                        write_json(study / ds / "pixel-cache-provenance.json", {"source": str(source_pixels.resolve()),
                            "sha256": file_hash(destination_pixels), "kind": "Original pixels and train-only normalization; no learned weights reused."})
                cache = make_cache(original / "data" / ds / "manifest.json", study / ds / "pixels.pt",
                                   configuration["resolution"], 40 if smoke else None)
                x = normalized(cache["splits"]["train"]["pixels"], cache, device)
                y = cache["splits"]["train"]["y"].to(device)
                vx = normalized(cache["splits"]["validation"]["pixels"], cache, device)
                vy = cache["splits"]["validation"]["y"].to(device)
                base = {**configuration, "classes": len(cache["classes"])}
                def execute(name, config):
                    progress(status="training", model=name)
                    record = run_model(study / ds / name, cache, x, y, vx, vy, config, name, progress)
                    records = export_results(study, output, configuration)
                    progress(completed_model_records=len(records))
                    return record
                if stage in ("main", "all"):
                    candidates = []
                    for architecture in ("pixel_tree", "hybrid"):
                        config = {**base, "architecture": architecture}
                        name = f"main-{architecture}-w{config['width']}"
                        record = execute(name, config)
                        candidates.append((name, config, record))
                        if record["train"]["accuracy"] < configuration["near_perfect"] and configuration.get("escalation"):
                            wider = {**config, **configuration["escalation"]}
                            name = f"fit-{architecture}-w{wider['width']}"
                            candidates.append((name, wider, execute(name, wider)))
                    # Same CNN family, trained as standalone binary roots on the full training split.
                    execute("cnn-only-roots", {**base, "architecture": "cnn_root", "max_depth": 0, "max_nodes": 1})
                    fitting = [t for t in candidates if t[2]["train"]["accuracy"] >= configuration["near_perfect"]]
                    selected = min(fitting, key=lambda t: t[2]["size"]["parameters"]) if fitting else max(candidates, key=lambda t: t[2]["train"]["accuracy"])
                    name, selected_config, selected_record = selected
                    roots = 1 if len(cache["classes"]) == 2 else len(cache["classes"])
                    matched_width = max(1, round((selected_record["size"]["parameters"] / roots - 4) / (3 * configuration["resolution"]**2 + 2)))
                    matched_config = {**selected_config, "architecture": "pixel_tree", "width": matched_width, "max_depth": 0, "max_nodes": 1}
                    if "root_width" in selected_config:
                        matched_config.update(root_width=matched_width, root_epochs=selected_config["mlp_epochs"])
                    matched = execute("parameter-matched-mlp-roots", matched_config)
                    write_json(study / ds / "selection.json", {"selected": name, "near_perfect_target_met": bool(fitting),
                              "selection": "Smallest near-fitting model, otherwise highest training accuracy; no test selection.",
                              "matched_parameter_difference": matched["size"]["parameters"] - selected_record["size"]["parameters"]})
                    progress(status="pruning", model=name)
                    prune(study / ds / "pruned", study / ds / name, cache, x, y, vx, vy, selected_config)
                    if configuration.get("report_fit_preserving_prune") and fitting:
                        prune(study / ds / "pruned-retain-fit", study / ds / name, cache, x, y, vx, vy,
                              {**selected_config, "preserve_training_fit_when_pruning": True})
                    narrow = {**selected_config, "width": max(1, selected_config["width"] // 2)}
                    execute("narrower-selected-architecture", narrow)
                    progress(status="visualizing", model=name)
                    from .visualize import node_visualizations
                    node_visualizations(study / ds / name, cache, x, y, output / ds)
                if stage in ("sweep", "all"):
                    for width in configuration["sweep_widths"]:
                        execute(f"capacity-pixel-tree-w{width}", {**base, "architecture": "pixel_tree", "width": width,
                                "max_depth": configuration["sweep_depth"], "max_nodes": configuration["sweep_nodes"]})
                del x, y, vx, vy, cache
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                verify_originals(snapshot)
            records = export_results(study, output, configuration)
            progress(status="complete", completed_model_records=len(records), original_results_verified_unchanged=True,
                     training_target_met={ds: read_json(study / ds / "selection.json")["near_perfect_target_met"]
                                          for ds in configuration["datasets"] if (study / ds / "selection.json").exists()})
            print("Version 2 declared stage complete; results:", output, flush=True)
            return status
        except Exception as exc:
            progress(status="failed", error=repr(exc))
            raise
