from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ell409-v2-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .common import load_checkpoint, write_json
from .train import scores_on


def plot_measurements(records, output):
    for ds in sorted({r["dataset"] for r in records}):
        local = [r for r in records if r["dataset"] == ds]
        mains = [r for r in local if not r["name"].startswith("capacity-")]
        if mains:
            fig, ax = plt.subplots(figsize=(10, max(4, .65 * len(mains))))
            positions = list(range(len(mains)))
            ax.barh([p-.18 for p in positions], [100*r["train"]["accuracy"] for r in mains], .35, label="Training")
            ax.barh([p+.18 for p in positions], [100*r["validation"]["accuracy"] for r in mains], .35, label="Validation")
            ax.set_yticks(positions, [r["name"] for r in mains]); ax.set_xlim(0, 100)
            resolution = mains[0]["config"]["resolution"]
            ax.set_xlabel("Accuracy (%)"); ax.set_title(f"{ds}: Version 2, original labels, {resolution}x{resolution} pixels")
            ax.legend(); fig.tight_layout(); fig.savefig(Path(output) / (ds + "-architecture-comparison.png"), dpi=170); plt.close(fig)
        sweep = sorted([r for r in local if r["name"].startswith("capacity-")], key=lambda r:r["config"]["width"])
        if sweep:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            widths = [r["config"]["width"] for r in sweep]
            for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
                axes[0].plot(widths, [100*(1-r[split]["accuracy"]) for r in sweep], "o-", color=color, label=split)
                axes[1].plot(widths, [r[split]["loss"] for r in sweep], "o-", color=color, label=split)
            axes[0].plot(widths, [100*(1-r["root_ablation"]["validation"]["accuracy"]) for r in sweep], "s--", color="tab:green", label="validation, root only")
            axes[0].set_ylabel("Classification error (%)"); axes[1].set_ylabel("Loss")
            first = next((r for r in sweep if r["train"]["errors"] == 0), None)
            for ax in axes:
                ax.set_xscale("log", base=2); ax.set_xticks(widths, [str(w) for w in widths])
                root_budget = sweep[0]["config"].get("root_width")
                ax.set_xlabel(f"Correction width; root fixed at {root_budget}" if root_budget is not None else "Node hidden width; adaptive correction topology")
                ax.legend(); ax.grid(alpha=.2)
                if first:
                    ax.axvline(first["config"]["width"], linestyle=":", color="gray")
            fig.suptitle(ds + ": exploratory clean-label capacity study")
            fig.tight_layout(); fig.savefig(Path(output) / (ds + "-capacity.png"), dpi=180); plt.close(fig)
            write_json(Path(output) / (ds + "-capacity-observations.json"), {
                "completed_widths": widths, "first_observed_zero_training_error_width": first["config"]["width"] if first else None,
                "actual_parameters": [r["size"]["parameters"] for r in sweep],
                "caution": "No forced pattern classification. Adaptive topology, a single seed, and finite optimization limit double-descent claims."})


def attribution(base, sample, kind):
    image = sample.detach().clone().requires_grad_(True)
    base.zero_grad(set_to_none=True)
    captured = []
    handle = None
    if kind == "cnn":
        def hook(module, inputs, output):
            output.retain_grad()
            captured.append(output)
        handle = base.trunk[3].register_forward_hook(hook)
    try:
        base(image).sum().backward()
        if kind == "cnn":
            activation = captured[0]
            weights = activation.grad.mean((2, 3), keepdim=True)
            heat = (weights * activation).sum(1).relu()[:, None]
            heat = F.interpolate(heat, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        else:
            heat = image.grad[0].abs().mean(0)
        return heat.detach().cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
        base.zero_grad(set_to_none=True)


def node_visualizations(folder, cache, x, y, output):
    from .study import load_model
    model, _ = load_model(folder, x.device)
    audit = load_checkpoint(Path(folder) / "audit.pt")["tasks"]
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    class_index = "1" if model.classes == 2 else "0"
    root = model.roots[0]
    available = list(root.walk())
    paths = [p for p in ("root", "root.A", "root.B", available[-1][0]) if any(a[0] == p for a in available)]
    paths = list(dict.fromkeys(paths))
    summaries = []
    indices = torch.arange(len(x), device=x.device)
    for path in paths:
        node = next(n for p, _, n in available if p == path)
        task = audit[class_index][path]
        scores = scores_on(node.base, x, indices)
        active = scores >= 0
        assigned_positive = task["indices"][task["labels"].bool()].to(x.device)
        strongest = scores.argsort(descending=True)[:2].tolist()
        if len(assigned_positive):
            ranked = assigned_positive[scores[assigned_positive].argsort(descending=True)].tolist()
            strongest.append(next((i for i in ranked if i not in strongest), ranked[0]))
        else:
            strongest.append(int(task["indices"][0]))
        corrected_active = scores_on(node, x, indices) >= 0
        summary = {"node": path, "kind": node.kind, "assigned_examples": len(task["labels"]),
                   "assigned_positive_examples": int(task["labels"].sum()),
                   "full_training_base_activation_frequency": float(active.float().mean()),
                   "full_training_corrected_activation_frequency": float(corrected_active.float().mean()),
                   "original_class_composition_of_base_activations": torch.bincount(y[active], minlength=model.classes).cpu().tolist(),
                   "selected_indices": strongest, "interpretation": "Continuous base-score sensitivity; not proof of a semantic concept."}
        summaries.append(summary)
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for row, index in enumerate(strongest):
            sample = x[index:index+1]
            raw = cache["splits"]["train"]["pixels"][index].permute(1, 2, 0).numpy()
            heat = attribution(node.base, sample, node.kind)
            baseline = float(scores[index])
            masked = sample.repeat(16, 1, 1, 1)
            size = sample.shape[-1] // 4
            for k in range(16):
                a, b = divmod(k, 4)
                masked[k, :, a*size:(a+1)*size, b*size:(b+1)*size] = 0
            with torch.no_grad():
                changes = (baseline - node.base(masked)).reshape(4, 4).cpu().numpy()
            axes[row, 0].imshow(raw)
            axes[row, 0].set_title(("Strong base response" if row < 2 else "Assigned positive correction" if path != "root" else "Assigned positive class") + f"\n{cache['classes'][int(y[index])]}, score {baseline:.2f}", fontsize=9)
            axes[row, 1].imshow(raw); axes[row, 1].imshow(heat, cmap="magma", alpha=.55)
            axes[row, 1].set_title("Grad-CAM" if node.kind == "cnn" else "Input-gradient sensitivity")
            bound = max(float(abs(changes).max()), 1e-8)
            mapping = axes[row, 2].imshow(changes, cmap="coolwarm", vmin=-bound, vmax=bound)
            axes[row, 2].set_title("Occlusion: base-score decrease", fontsize=10)
            fig.colorbar(mapping, ax=axes[row, 2], shrink=.8, pad=.02)
            for ax in axes[row]: ax.axis("off")
        fig.suptitle(f"{path}, {node.kind}: trained on {len(task['labels'])} assigned examples")
        fig.tight_layout(); fig.savefig(output / (path + "-responses.png"), dpi=170); plt.close(fig)
    write_json(output / "node-observations.json", summaries)
