"""True uninterrupted epoch trajectories for representative root-only node models."""
from pathlib import Path

import torch

from .common import digest, environment, load_checkpoint, seed_everything, write_json
from .models import BaseNode
from .train import fit_binary


def root_epoch_history(run, device, epochs=300, widths=(8, 64, 512), seed=409):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    bundle = load_checkpoint(run / "features.pt")
    positive = 1 if bundle["classes"] == 2 else bundle.get("class_names", []).index("cat")
    feature = bundle["splits"]
    x, y = feature["train"]["x"].to(device), (feature["train"]["y"] == positive).float().to(device)
    vx, vy = feature["validation"]["x"].to(device), (feature["validation"]["y"] == positive).float().to(device)
    folder = run / "root-epoch-history"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for width in widths:
        seed_everything(seed)
        model = BaseNode(x.shape[1], width).to(device)
        config = {"node_epochs": epochs, "node_lr": .001, "feature_batch_size": 1024}
        identity = {"protocol": "root-only binary node, before correction children", "class": positive,
                    "width": width, "seed": seed, "features": bundle["fingerprint"]}
        stats = fit_binary(model, x, y, folder / f"w{width}.pt", config, identity, validation=(vx, vy))
        record = {"width": width, "positive_class": positive, "positive_name": bundle["class_names"][positive],
                  "seed": seed, "environment": environment(), "protocol": identity["protocol"],
                  "features": bundle["fingerprint"], **stats}
        write_json(folder / f"w{width}.json", record)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        history = stats["history"]
        axis = [r["epoch"] for r in history]
        axes[0].plot(axis, [r["errors"] / r["examples"] for r in history], label="Train")
        axes[0].plot(axis, [r["validation"]["errors"] / r["validation"]["examples"] for r in history], label="Validation")
        axes[1].plot(axis, [r["loss"] for r in history], label="Train")
        axes[1].plot(axis, [r["validation"]["loss"] for r in history], label="Validation")
        first = next((r["epoch"] for r in history if r["errors"] == 0), None)
        for ax in axes:
            ax.set_xlabel("Actual training epoch of root-only binary node")
            ax.legend(); ax.grid(alpha=.2)
            if first is not None: ax.axvline(first, color="gray", linestyle="--")
        axes[0].set_ylabel("Classification error"); axes[1].set_ylabel("Unweighted BCE")
        fig.suptitle(f"Root-only width {width}: {record['positive_name']} versus remaining classes")
        fig.tight_layout()
        (run / "figures").mkdir(exist_ok=True)
        fig.savefig(run / "figures" / f"root-epochs-w{width}.png", dpi=150); plt.close(fig)
        records.append({"width": width, "positive_name": record["positive_name"], "first_zero_error_epoch": first,
                        "best_training_errors": stats["best_errors"], "last_validation": history[-1]["validation"]})
        print(f"Root-only epoch trace complete: width {width}, {epochs} epochs, best errors {stats['best_errors']}", flush=True)
    write_json(run / "root-epoch-summary.json", {"models": records,
               "protocol": "Binary root-only trajectories, not final multiclass correction-tree trajectories; validation never selects weights.",
               "uses_test_labels": False})
    return records
