from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import load_checkpoint, read_json, write_json
from .data import ImageSplit
from .models import SharedCNN
from .train import load_tree


def grad_cam(cnn, node, image, mean, std):
    """Differentiate the continuous base score, with gradients through the frozen trunk."""
    x = image[None].detach().requires_grad_(True)
    spatial = cnn.spatial_features(x)
    features = (spatial.mean((2, 3)) - mean) / std
    score = node.base(features)[0]
    gradient, = torch.autograd.grad(score, spatial)
    weights = gradient.mean((2, 3), keepdim=True)
    cam = (weights * spatial).sum(1, keepdim=True).relu()
    cam = F.interpolate(cam, image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    maximum = cam.max()
    if maximum > 0:
        cam = cam / maximum
    return cam.detach().cpu().numpy(), float(score.detach())


def occlusion_map(cnn, node, image, mean, std, grid=4):
    height, width = image.shape[-2:]
    scores, positions, images = [], [], []
    with torch.no_grad():
        original = float(node.base((cnn.features(image[None]) - mean) / std)[0])
        for r in range(grid):
            for c in range(grid):
                y0, y1 = height * r // grid, height * (r + 1) // grid
                x0, x1 = width * c // grid, width * (c + 1) // grid
                masked = image.clone()
                masked[:, y0:y1, x0:x1] = 0  # training mean in normalized image space
                images.append(masked); positions.append((y0, y1, x0, x1))
        features = (cnn.features(torch.stack(images)) - mean) / std
        delta = original - node.base(features)
    output = np.zeros((height, width), dtype=np.float32)
    for value, (y0, y1, x0, x1) in zip(delta.cpu().tolist(), positions):
        output[y0:y1, x0:x1] = value
    return output


def visualize_nodes(manifest_path, run, device, max_nodes=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    selected = read_json(run / "selected-tree.json")["name"]
    model, saved = load_tree(run / selected / "tree.pt", device)
    bundle = load_checkpoint(run / "features.pt")
    manifest = read_json(manifest_path)
    cnn = SharedCNN(len(manifest["classes"])).to(device)
    cnn.load_state_dict(load_checkpoint(run / "cnn-best.pt")["model"])
    cnn.eval()
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)
    mean, std = bundle["feature_mean"].to(device), bundle["feature_std"].to(device)
    data = ImageSplit(manifest_path, "train")
    features, labels = bundle["splits"]["train"]["x"], bundle["splits"]["train"]["y"]
    candidates = [(i, path, depth, node) for i, root in enumerate(model.roots)
                  for path, depth, node in root.walk() if node.active]
    chosen = [candidates[0]]
    corrections = [c for c in candidates if c[2] > 0]
    # Cover shallow and deep corrections rather than choosing all nodes from one level.
    corrections.sort(key=lambda c: (c[2], c[0], c[1]))
    if corrections:
        chosen.extend([corrections[0]])
        chosen.extend(sorted(corrections[1:], key=lambda c: (-c[2], c[0], c[1]))[:max_nodes - 2])
    output = run / "figures" / "nodes"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for root_index, path, depth, node in chosen:
        key = str(1 if model.classes == 2 else root_index)
        task = saved["tasks"][key][path]
        raw_scores, activations = [], []
        with torch.no_grad():
            for x in features.split(1024):
                raw_scores.append(node.base(x.to(device)).cpu())
                activations.append((node(x.to(device)) >= 0).cpu())
        raw_scores, activated = torch.cat(raw_scores), torch.cat(activations)
        best = raw_scores.argsort(descending=True)[:2].tolist()
        task_ids = task["indices"].long()
        positive = task_ids[task["labels"].bool()]
        positive_order = positive[raw_scores[positive].argsort(descending=True)] if len(positive) else positive
        shown = best + (positive_order[:1].tolist() if len(positive_order) else best[:1])
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        image_mean = data.mean.permute(1, 2, 0).numpy()
        image_std = data.std.permute(1, 2, 0).numpy()
        for column, index in enumerate(shown):
            image, label, _ = data[index]
            rgb = (image.permute(1, 2, 0).numpy() * image_std + image_mean).clip(0, 1)
            axes[0, column].imshow(rgb)
            prefix = "Strong response" if column < 2 else "Positive task target"
            axes[0, column].set_title(f"{prefix}: {manifest['classes'][label]}\nbase score {raw_scores[index]:.2f}")
            axes[0, column].axis("off")
        index = shown[0]
        image = data[index][0].to(device)
        rgb = (image.cpu().permute(1, 2, 0).numpy() * image_std + image_mean).clip(0, 1)
        cam, score = grad_cam(cnn, node, image, mean, std)
        occlusion = occlusion_map(cnn, node, image, mean, std)
        axes[1, 0].imshow(rgb); axes[1, 0].imshow(cam, cmap="jet", alpha=.45, vmin=0, vmax=1)
        axes[1, 0].set_title("Grad-CAM: continuous base score"); axes[1, 0].axis("off")
        magnitude = max(float(np.abs(occlusion).max()), 1e-6)
        heat = axes[1, 1].imshow(occlusion, cmap="coolwarm", vmin=-magnitude, vmax=magnitude)
        axes[1, 1].set_title("Occlusion: original minus masked score"); axes[1, 1].axis("off")
        fig.colorbar(heat, ax=axes[1, 1], fraction=.046)
        class_counts = torch.bincount(labels[activated], minlength=len(manifest["classes"])).tolist()
        axes[1, 2].bar(range(len(class_counts)), class_counts)
        axes[1, 2].set_xticks(range(len(class_counts)), manifest["classes"], rotation=65, fontsize=7)
        axes[1, 2].set_title("Classes activating corrected node")
        fig.suptitle(f"Tree {key}, {path}, depth {depth}; all examples from training split")
        fig.tight_layout()
        figure_name = f"class-{key}-{path.replace('.', '-')}.png"
        fig.savefig(output / figure_name, dpi=140, bbox_inches="tight", pad_inches=.15); plt.close(fig)
        summaries.append({"tree": key, "path": path, "depth": depth, "task_examples": len(task_ids),
                          "task_positive_examples": len(positive), "final_task_errors": task["final_errors"],
                          "activation_frequency": float(activated.float().mean()), "class_counts": class_counts,
                          "example_ids": [data.records[i]["id"] for i in shown], "figure": figure_name,
                          "cam_nonzero": bool(cam.max() > 0), "base_score": score,
                          "occlusion_max_absolute_change": magnitude})
    write_json(run / "node-observations.json", {"nodes": summaries,
               "interpretation": "Maps show empirical sensitivity, not proof of semantic concepts.",
               "last_spatial_feature_receptive_field_pixels": 36,
               "global_average_pooling": "Node base scores can depend on the full input image."})
    return summaries
