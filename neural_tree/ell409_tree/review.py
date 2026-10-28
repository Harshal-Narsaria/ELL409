"""Read-only analysis of locked experiments; no model selection using test labels."""
from pathlib import Path
import csv
import json

from .common import digest, read_json, write_json


def verify_test_records(run, audit, config):
    run = Path(run)
    candidates = sorted(p.parent.name for p in run.glob("*/metrics.json"))
    declaration = {"manifest": audit["fingerprint"], "candidates": candidates, "configuration": digest(config)}
    if read_json(run / "test-declaration.json") != declaration:
        raise ValueError("Published candidate list differs from the locked test declaration")
    for name in candidates:
        model = read_json(run / name / "metrics.json")
        test = read_json(run / name / "test-metrics.json")
        if model["manifest"] != audit["fingerprint"] or test["declaration"] != digest(declaration):
            raise ValueError(f"Experiment identity mismatch: {name}")
        for split, metric in [("train", model["train"]), ("validation", model["validation"]), ("test", test)]:
            if metric["examples"] != audit["counts"][split]:
                raise ValueError(f"Incomplete evaluation: {name}/{split}")
            if not 0 <= metric["errors"] <= metric["examples"] or abs(metric["accuracy"] - (1 - metric["errors"] / metric["examples"])) > 1e-12:
                raise ValueError(f"Inconsistent error count: {name}/{split}")
    return candidates


def collect_review(runs_root):
    review = {"datasets": {}, "selection_policy": "Selected models were fixed using training records before test evaluation. Smallest-model searches below use training/validation only; they do not replace that selection."}
    for dataset in ("cifar10", "cats_vs_dogs"):
        run = Path(runs_root) / dataset
        if not (run / "study-complete.json").exists():
            raise ValueError(f"Study not complete: {dataset}")
        audit = read_json(run / "data-audit.json")
        config = read_json(run / "configuration.json")["config"]
        names = verify_test_records(run, audit, config)
        models = []
        for name in names:
            metric = read_json(run / name / "metrics.json")
            models.append({**metric, "test": read_json(run / name / "test-metrics.json")})
        selected = read_json(run / "selected-tree.json")["name"]
        chosen = next(r for r in models if r["name"] == selected)
        if chosen["train"]["accuracy"] < config["near_perfect"]:
            raise ValueError(f"Selected model misses fitting target: {dataset}")
        roots = next(r for r in models if r["name"] == "roots-only")
        pruned = next(r for r in models if r["name"] == "pruned")
        if pruned["validation"]["accuracy"] + 1e-12 < chosen["validation"]["accuracy"] - .005:
            raise ValueError(f"Pruning violates cumulative validation tolerance: {dataset}")
        smallest_fit = min((r for r in models if r["train"]["accuracy"] >= config["near_perfect"]), key=lambda r: (r["size"]["active_parameters"], r["name"]))
        smallest_val = min((r for r in models if r["validation"]["accuracy"] >= chosen["validation"]["accuracy"] - .005), key=lambda r: (r["size"]["active_parameters"], r["name"]))
        node_notes = read_json(run / "node-interpretations.json") if (run / "node-interpretations.json").exists() else None
        if node_notes is None or node_notes["model"] != selected:
            raise ValueError(f"Reviewed node interpretations missing for selected model: {dataset}")
        epochs = []
        for width in (8, 64, 512):
            item = read_json(run / "root-epoch-history" / f"w{width}.json")
            history = item["history"]
            if len(history) != 300 or any(r["epoch"] != i + 1 or "validation" not in r for i, r in enumerate(history)):
                raise ValueError(f"Incomplete actual epoch trajectory: {dataset}/{width}")
            epochs.append({"width": width, "positive_name": item["positive_name"],
                "first_zero_error_epoch": next((r["epoch"] for r in history if r["errors"] == 0), None),
                "first": history[0], "last": history[-1],
                "minimum_validation_errors": min(r["validation"]["errors"] for r in history),
                "note": "Actual last-epoch trajectory, not validation-selected weights or the final recursive multiclass model."})
        summary = {"audit": audit, "configuration": config, "models": models, "selected": chosen,
            "cnn": {**read_json(run / "cnn-baseline.json"), "test": read_json(run / "cnn-test.json")},
            "epoch_models": epochs, "node_interpretations": node_notes,
            "sweeps": {tag: read_json(run / f"{tag}-interpretation.json") for tag in ("sweep", "sweep-long")},
            "smallest_near_perfect": smallest_fit["name"], "smallest_validation_tolerance": smallest_val["name"],
            "compression": {"node_reduction": chosen["size"]["active_nodes"] - pruned["size"]["active_nodes"],
                "parameter_reduction_fraction": 1 - pruned["size"]["active_parameters"] / chosen["size"]["active_parameters"],
                "file_reduction_fraction": 1 - pruned["deployment_bytes"] / chosen["deployment_bytes"],
                "pruned_trunk_fraction": pruned["size"]["trunk_parameters"] / pruned["size"]["active_parameters"],
                "narrower_nodes": {r["name"]: r["size"]["active_nodes"] for r in models if r["name"].startswith("narrow")}},
            "correction_delta": {s: chosen[s]["accuracy"] - roots[s]["accuracy"] for s in ("train", "validation", "test")},
            "flagged_optimization_runs": [r["name"] for r in models if r.get("convergence", {}).get("needs_extension")],
            "candidate_lock_verified": True, "completed_model_configurations": len(models)}
        review["datasets"][dataset] = summary
    return review


def write_review(runs_root, output_dir="results"):
    review = collect_review(runs_root)
    output = Path(output_dir)
    write_json(output / "analysis.json", review)
    rows, binary = [], []
    for dataset, data in review["datasets"].items():
        for model in data["models"]:
            rows.append({"dataset": dataset, "model": model["name"], "width": model["config"]["width"], "seed": model["config"]["seed"],
                "epochs": model["config"]["node_epochs"], **{f"{s}_{k}": model[s][k] for s in ("train", "validation", "test") for k in ("examples", "errors", "accuracy", "loss")},
                **model["size"], "deployment_bytes": model["deployment_bytes"], "training_seconds": model.get("training_seconds")})
        selected = data["selected"]
        for split in ("train", "validation", "test"):
            for i, diagnostic in enumerate(selected[split]["binary_trees"]):
                label = data["audit"]["classes"][1 if len(data["audit"]["classes"]) == 2 else i]
                binary.append({"dataset": dataset, "model": selected["name"], "split": split, "positive_class": label, **diagnostic})
    for name, records in [("model-comparison.csv", rows), ("binary-diagnostics.csv", binary)]:
        with (output / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    text = ["# Completed experimental results", "", "Both datasets meet the designated training-fit target. Test candidates were locked before held-out evaluation; no model is selected here using test accuracy.", "",
        "| Dataset | Selected training accuracy | Validation accuracy | Test accuracy | Training errors |", "|---|---:|---:|---:|---:|"]
    for ds, data in review["datasets"].items():
        r = data["selected"]
        text.append(f"| {ds} | {r['train']['accuracy']:.4%} | {r['validation']['accuracy']:.2%} | {r['test']['accuracy']:.2%} | {r['train']['errors']} / {r['train']['examples']} |")
    text += ["", "## Conclusions", "", "No clear model-capacity double descent was observed in either dataset in the tested range. CIFAR validation error generally falls with capacity; Cats vs Dogs shows small, irregular fluctuations. This is evidence about these experiments, not a claim that double descent cannot occur.", "",
        "Epoch histories show that very low training error can coexist with rising held-out BCE. Confidence on incorrectly classified held-out examples can worsen even when classification error changes little. Binary root-only trajectories must not be interpreted as global CIFAR multiclass trajectories.", ""]
    for ds, data in review["datasets"].items():
        c = data["compression"]
        text += [f"## {ds}", "", f"The selected tree has {data['selected']['size']['active_nodes']} active nodes. Pruning leaves {'10 roots' if ds == 'cifar10' else 'one root'}, reduces total parameters by {c['parameter_reduction_fraction']:.2%}, and preserves the selected validation accuracy in these measurements.", "",
            f"Smallest tested active model retaining at least 99.5% training fit: `{data['smallest_near_perfect']}`. Smallest within the original validation tolerance: `{data['smallest_validation_tolerance']}`. These searches use train/validation records only and do not change the predeclared selected model.", "",
            "Narrower nodes can require more correction nodes: reducing width is not a guarantee of a smaller or faster complete classifier. The shared CNN is counted once and dominates the most compact models.", ""]
        text += [p + "\n" for p in data["node_interpretations"]["observations"]]
    text += ["## Limitations", "", "Only selected widths have three seeds. Inactive allocated correction positions do not contribute to inference; active topology changes as error-driven growth responds to capacity. Remaining optimization flags limit strong capacity claims. Timing measurements exclude disk reads and resizing. Attribution maps indicate empirical sensitivity, not proven semantic concepts.", "", "## Submission status", "", "Scientific measurements and interpretations are complete. The private GitHub URL and instructor-access verification must be added after repository creation."]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    plot_review(review, output)
    print(f"Reviewed {sum(s['completed_model_configurations'] for s in review['datasets'].values())} locked model configurations across both datasets.")
    return review


def plot_review(review, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    names = ["roots-only", "adaptive-w64", "pruned", "narrow-w32", "narrow-w16"]
    labels = ["Roots only", "Correction tree", "Pruned", "Width 32", "Width 16"]
    for dataset, data in review["datasets"].items():
        records = [next(r for r in data["models"] if r["name"] == n) for n in names]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(len(records))
        trunk = [r["size"]["trunk_parameters"] for r in records]
        nodes = [r["size"]["active_parameters"] - r["size"]["trunk_parameters"] for r in records]
        axes[0].bar(x, trunk, label="Shared CNN")
        axes[0].bar(x, nodes, bottom=trunk, label="Tree networks")
        for i, r in enumerate(records): axes[0].text(i, r["size"]["active_parameters"] + 4000, f"{r['size']['active_nodes']} nodes", ha="center", fontsize=8)
        axes[0].set_ylabel("Actual parameters, CNN counted once")
        axes[0].set_ylim(0, max(r["size"]["active_parameters"] for r in records) * 1.18)
        for split in ("train", "validation", "test"):
            axes[1].plot(x, [r[split]["accuracy"] * 100 for r in records], marker="o", label=split)
        axes[1].set_ylabel("Accuracy (%)")
        for ax in axes:
            ax.set_xticks(x, labels, rotation=20, ha="right"); ax.legend(); ax.grid(axis="y", alpha=.2)
        fig.suptitle(dataset + ": compression trades node width against tree growth")
        fig.tight_layout(); fig.savefig(folder / f"{dataset}-compression.png", dpi=170, bbox_inches="tight"); plt.close(fig)
        confusion = np.asarray(data["selected"]["test"]["confusion"])
        percentage = confusion / confusion.sum(axis=1, keepdims=True) * 100
        classes = data["audit"]["classes"]
        fig, ax = plt.subplots(figsize=(7, 5.7))
        image = ax.imshow(percentage, cmap="Blues", vmin=0, vmax=100)
        for r in range(len(classes)):
            for c in range(len(classes)):
                ax.text(c, r, f"{percentage[r,c]:.1f}", ha="center", va="center", fontsize=8, color="white" if percentage[r,c] > 55 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set(xlabel="Predicted class", ylabel="True class", title=dataset + ": selected model test confusion, row percentages")
        fig.colorbar(image, ax=ax, label="% of true-class examples")
        fig.tight_layout(); fig.savefig(folder / f"{dataset}-confusion.png", dpi=160, bbox_inches="tight"); plt.close(fig)
