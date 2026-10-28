from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import (digest, load_checkpoint, read_json, restore_rng, rng_state,
                     save_checkpoint, seed_everything, state_hash, tensor_hash, write_json)
from .data import ImageSplit
from .models import BaseNode, CorrectionNode, SharedCNN, TreeClassifier, correction_targets


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def image_metrics(model, loader, device):
    model.eval()
    errors, loss, count = 0, 0., 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            errors += int((scores.argmax(1) != y).sum())
            loss += float(F.cross_entropy(scores, y, reduction="sum"))
            count += len(y)
    return {"examples": count, "errors": errors, "accuracy": 1 - errors / count,
            "loss": loss / count}


def train_cnn(manifest_path, run_dir, config, device):
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path)
    seed_everything(config["seed"])
    model = SharedCNN(len(manifest["classes"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["cnn_lr"], weight_decay=0)
    train = ImageSplit(manifest_path, "train")
    validation = ImageSplit(manifest_path, "validation")
    kwargs = {"batch_size": config["image_batch_size"], "num_workers": config.get("workers", 0),
              "pin_memory": device.type == "cuda"}
    training_loader = DataLoader(train, shuffle=True, **kwargs)
    train_eval = DataLoader(train, shuffle=False, **kwargs)
    validation_loader = DataLoader(validation, shuffle=False, **kwargs)
    signature = digest({"manifest": manifest["fingerprint"], "seed": config["seed"],
                        "batch": config["image_batch_size"], "lr": config["cnn_lr"]})
    latest = run / "cnn-latest.pt"
    history, start, best_key, best_state, elapsed = [], 0, (float("inf"), float("inf")), None, 0.
    if latest.exists():
        saved = load_checkpoint(latest)
        if saved["signature"] != signature:
            raise ValueError("CNN resume configuration or dataset changed. Use a new run directory.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, start = saved["history"], saved["epoch"]
        best_key, best_state, elapsed = tuple(saved["best_key"]), saved["best_model"], saved["elapsed"]
    for epoch in range(start, config["cnn_epochs"]):
        before = time.perf_counter()
        model.train()
        for x, y, _ in training_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CNN loss")
            loss.backward()
            optimizer.step()
        tr = image_metrics(model, train_eval, device)
        va = image_metrics(model, validation_loader, device)
        seconds = time.perf_counter() - before
        elapsed += seconds
        history.append({"epoch": epoch + 1, "train": tr, "validation": va, "seconds": seconds})
        key = (tr["errors"], tr["loss"])
        if key < best_key:
            best_key, best_state = key, cpu_state(model)
        payload = {"signature": signature, "model": cpu_state(model), "best_model": best_state,
                   "best_key": best_key, "optimizer": optimizer.state_dict(), "rng": rng_state(),
                   "epoch": epoch + 1, "history": history, "elapsed": elapsed}
        save_checkpoint(latest, payload)
        if epoch + 1 == config.get("early_cnn_epoch", 5):
            save_checkpoint(run / "cnn-early.pt", {"model": cpu_state(model), "epoch": epoch + 1,
                            "manifest": manifest["fingerprint"]})
        write_json(run / "cnn-history.json", history)
        print(f"CNN epoch {epoch + 1}: train {tr['accuracy']:.4%}, validation {va['accuracy']:.4%}, {seconds:.1f}s", flush=True)
    model.load_state_dict(best_state)
    save_checkpoint(run / "cnn-best.pt", {"model": best_state, "manifest": manifest["fingerprint"],
                    "classes": len(manifest["classes"]), "elapsed": elapsed})
    return model


def cache_features(manifest_path, cnn_path, output_path, batch_size, device, workers=0):
    manifest = read_json(manifest_path)
    model = SharedCNN(len(manifest["classes"])).to(device)
    checkpoint = load_checkpoint(cnn_path)
    if checkpoint["manifest"] != manifest["fingerprint"]:
        raise ValueError("CNN and split manifest differ")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trunk = nn.Sequential(*model.blocks)
    trunk_hash = state_hash(trunk)
    deployment_metadata = {"trunk_state": cpu_state(model.blocks), "image_mean": manifest["normalization"]["mean"],
                           "image_std": manifest["normalization"]["std"], "resolution": manifest["resolution"],
                           "class_names": manifest["classes"]}
    if Path(output_path).exists():
        old = load_checkpoint(output_path)
        if old["manifest"] != manifest["fingerprint"] or old["trunk_hash"] != trunk_hash:
            raise ValueError("Feature cache is stale; use a new output path")
        if "trunk_state" not in old:
            old.update(deployment_metadata)
            save_checkpoint(output_path, old)
        return old
    bundle = {"manifest": manifest["fingerprint"], "trunk_hash": trunk_hash,
              "trunk_parameters": sum(p.numel() for p in model.blocks.parameters()),
              "cnn_path": Path(cnn_path).name, "classes": len(manifest["classes"]), "splits": {}, **deployment_metadata}
    for split in ("train", "validation", "test"):
        loader = DataLoader(ImageSplit(manifest_path, split), batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda")
        xs, ys = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                xs.append(model.features(x.to(device)).cpu())
                ys.append(y)
        bundle["splits"][split] = {"x": torch.cat(xs), "y": torch.cat(ys)}
    raw_train = bundle["splits"]["train"]["x"]
    mean, std = raw_train.mean(0), raw_train.std(0, unbiased=False).clamp_min(1e-6)
    bundle["feature_mean"], bundle["feature_std"] = mean, std
    for split in bundle["splits"].values():
        split["x"] = (split["x"] - mean) / std
    import numpy as np
    from collections import defaultdict, Counter
    identical = defaultdict(Counter)
    for row, label in zip(bundle["splits"]["train"]["x"].numpy(), bundle["splits"]["train"]["y"].tolist()):
        identical[row.tobytes()][label] += 1
    unavoidable = sum(sum(labels.values()) - max(labels.values()) for labels in identical.values())
    bundle["feature_diagnostics"] = {"unique_training_features": len(identical),
                                    "conflicting_feature_groups": sum(len(labels) > 1 for labels in identical.values()),
                                    "minimum_errors_from_exact_feature_collisions": unavoidable,
                                    "training_accuracy_upper_bound": 1 - unavoidable / len(raw_train)}
    bundle["fingerprint"] = tensor_hash(bundle["splits"]["train"]["x"], bundle["splits"]["train"]["y"])
    save_checkpoint(output_path, bundle)
    write_json(Path(output_path).with_suffix(".diagnostics.json"), bundle["feature_diagnostics"])
    return bundle


def binary_metrics(model, x, y):
    model.eval()
    with torch.no_grad():
        scores = model(x)
        return int(((scores >= 0) != y.bool()).sum()), float(F.binary_cross_entropy_with_logits(scores, y.float()))


def fit_binary(model, x, y, checkpoint, config, identity, projection=None, validation=None):
    """Epoch-boundary exact resume; choose best by errors, then unweighted BCE."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["node_lr"])
    signature_data = {"identity": identity, "data": tensor_hash(x, y), "lr": config["node_lr"],
                      "batch": config["feature_batch_size"]}
    if validation is not None:
        signature_data["validation"] = tensor_hash(*validation)
    signature = digest(signature_data)
    start, history = 0, []
    best_key, best_model = binary_metrics(model, x, y), cpu_state(model)
    elapsed = 0.
    if Path(checkpoint).exists():
        saved = load_checkpoint(checkpoint)
        if saved["signature"] != signature:
            raise ValueError("Node resume data/configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, history = saved["epoch"], saved["history"]
        best_key, best_model = tuple(saved["best_key"]), saved["best_model"]
        elapsed = saved["elapsed"]
    positives, negatives = float(y.sum()), float(len(y) - y.sum())
    positive_weight = torch.tensor(negatives / positives if positives and negatives else 1., device=x.device)
    for epoch in range(start, config["node_epochs"]):
        before = time.perf_counter()
        model.train()
        order = torch.randperm(len(y), device=x.device)
        for indices in order.split(config["feature_batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[indices]), y[indices].float(), pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite node loss")
            loss.backward()
            optimizer.step()
            if projection:
                projection()
        errors, loss = binary_metrics(model, x, y)
        elapsed += time.perf_counter() - before
        row = {"epoch": epoch + 1, "errors": errors, "examples": len(y), "loss": loss}
        if validation is not None:
            vx, vy = validation
            val_errors, val_loss = binary_metrics(model, vx, vy)
            with torch.no_grad():
                predicted = model(vx) >= 0
                wanted = vy.bool()
                recall = float((predicted & wanted).sum()) / max(int(wanted.sum()), 1)
                specificity = float((~predicted & ~wanted).sum()) / max(int((~wanted).sum()), 1)
            row["validation"] = {"errors": val_errors, "examples": len(vy), "loss": val_loss,
                                 "positive_recall": recall, "negative_accuracy": specificity}
        history.append(row)
        if (errors, loss) < best_key:
            best_key, best_model = (errors, loss), cpu_state(model)
        save_checkpoint(checkpoint, {"signature": signature, "epoch": epoch + 1, "model": cpu_state(model),
                        "optimizer": optimizer.state_dict(), "rng": rng_state(), "best_key": best_key,
                        "best_model": best_model, "history": history, "elapsed": elapsed})
    model.load_state_dict(best_model)
    return {"history": history, "best_errors": best_key[0], "best_loss": best_key[1], "elapsed": elapsed}


class CorrectionFit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(3))

    def forward(self, inputs):
        # Columns: fixed base score, A decision, B decision.
        return inputs[:, 0] + self.weights[0] * inputs[:, 1] - self.weights[1] * inputs[:, 2] + self.weights[2]

    def project(self):
        with torch.no_grad():
            self.weights[:2].clamp_(min=0)


def exact_correction_candidate(inputs, target):
    """A 3-variable LP: eight worst-case groups suffice for all margin constraints.

    This preserves the PDF's signed parent refit and avoids an N-slack-variable LP.
    Return a margin-one solution when those fixed-score constraints are feasible.
    Infeasibility does not rule out a correct refit with a smaller margin, or a
    better solution after learning different base scores/child decisions.
    """
    import numpy as np
    from scipy.optimize import linprog
    values = inputs.detach().cpu().numpy().astype(np.float64)
    targets = target.detach().cpu().numpy()
    rows, bounds = [], []
    for label in (0, 1):
        sign = 2 * label - 1
        for a in (0, 1):
            for b in (0, 1):
                mask = (targets == label) & (values[:, 1] == a) & (values[:, 2] == b)
                if not mask.any():
                    continue
                rows.append([-sign * a, sign * b, -sign])
                bounds.append(float((sign * values[mask, 0]).min()) - 1)
    solved = linprog([1., 1., 0.], A_ub=np.array(rows), b_ub=np.array(bounds),
                     bounds=[(0, None), (0, None), (None, None)], method="highs")
    if not solved.success or not np.isfinite(solved.x).all():
        return None
    return torch.tensor(solved.x, dtype=inputs.dtype, device=inputs.device)


class TreeTrainer:
    def __init__(self, folder, config, features, seed, device):
        self.folder, self.config = Path(folder), config
        self.folder.mkdir(parents=True, exist_ok=True)
        self.features, self.seed, self.device = features, seed, device
        self.used = 0
        self.tasks = {}

    def grow(self, x, y, indices, path="root", depth=0):
        self.used += 1
        local_seed = self.seed + int(digest(path)[:7], 16)
        seed_everything(local_seed)
        node = CorrectionNode(self.features, self.config["width"]).to(self.device)
        final_path = self.folder / (path + "-final.pt")
        identity = {"path": path, "depth": depth, "width": self.config["width"],
                    "seed": self.seed, "max_depth": self.config["max_depth"],
                    "max_nodes": self.config["max_nodes"], "epochs": self.config["node_epochs"],
                    "lr": self.config["node_lr"], "batch": self.config["feature_batch_size"]}
        signature = digest({"identity": identity, "task": tensor_hash(x, y, indices)})
        if final_path.exists():
            saved = load_checkpoint(final_path)
            if saved["signature"] != signature:
                raise ValueError("Completed node task changed; use a new experiment directory")
            node = CorrectionNode.from_spec(saved["spec"]).to(self.device)
            node.load_state_dict(saved["model"])
            self.used += sum(1 for _ in node.walk()) - 1
            self.tasks.update(saved["tasks"])
            return node
        base_stats = fit_binary(node.base, x, y, self.folder / (path + "-base.pt"), self.config, identity)
        with torch.no_grad():
            scores = node.base(x)
            predictions = scores >= 0
        before_errors, before_loss = binary_metrics(node, x, y)
        self.tasks[path] = {"indices": indices.cpu(), "labels": y.cpu(),
                            "base_predictions": predictions.cpu(), "base": base_stats}
        targets = correction_targets(predictions, y)
        if depth < self.config["max_depth"] and before_errors:
            for side in ("A", "B"):
                child_y, mask = targets[side]
                if child_y[mask].sum() == 0 or self.used >= self.config["max_nodes"]:
                    continue
                child = self.grow(x[mask], child_y[mask], indices[mask], path + "." + side, depth + 1)
                setattr(node, side, child)
        if node.A is not None or node.B is not None:
            with torch.no_grad():
                a = (node.A(x) >= 0).float() if node.A is not None else torch.zeros_like(y)
                b = (node.B(x) >= 0).float() if node.B is not None else torch.zeros_like(y)
                inputs = torch.stack([scores, a, b], dim=1)
            correction = CorrectionFit().to(self.device)
            correction_config = {**self.config, "node_epochs": self.config.get("correction_epochs", 150),
                                 "node_lr": .05, "feature_batch_size": max(len(y), 1)}
            stats = fit_binary(correction, inputs, y, self.folder / (path + "-correction.pt"),
                               correction_config, {**identity, "stage": "correction"}, correction.project)
            exact = exact_correction_candidate(inputs, y)
            stats["margin_one_signed_refit_feasible"] = exact is not None
            if exact is not None:
                old = correction.weights.detach().clone()
                old_key = binary_metrics(correction, inputs, y)
                with torch.no_grad():
                    correction.weights.copy_(exact)
                if binary_metrics(correction, inputs, y) > old_key:
                    with torch.no_grad():
                        correction.weights.copy_(old)
            with torch.no_grad():
                node.alpha.copy_(correction.weights[0])
                node.beta.copy_(correction.weights[1])
                node.offset.copy_(correction.weights[2])
            after = binary_metrics(node, x, y)
            if after > (before_errors, before_loss):
                # Reject harmful refits while preserving the learned child tasks for auditing.
                with torch.no_grad():
                    node.alpha.zero_(); node.beta.zero_(); node.offset.zero_()
            self.tasks[path]["correction"] = stats
        after = binary_metrics(node, x, y)
        self.tasks[path]["final_errors"] = after[0]
        own_tasks = {p: task for p, task in self.tasks.items() if p == path or p.startswith(path + ".")}
        save_checkpoint(final_path, {"signature": signature, "spec": node.spec(), "model": cpu_state(node),
                        "tasks": own_tasks})
        print(f"{path}: {len(y)} examples, errors {before_errors} -> {after[0]}, depth {depth}", flush=True)
        return node


def train_trees(bundle, folder, config, device):
    x = bundle["splits"]["train"]["x"].to(device)
    labels = bundle["splits"]["train"]["y"].to(device)
    roots, tasks = [], {}
    start = time.perf_counter()
    classes = bundle["classes"]
    for c in ([1] if classes == 2 else range(classes)):
        targets = (labels == c).float()
        trainer = TreeTrainer(Path(folder) / f"class-{c}", config, x.shape[1], config["seed"] + c, device)
        root = trainer.grow(x, targets, torch.arange(len(labels), device=device))
        if config.get("fixed_topology"):
            root.pad(config["max_depth"])
        roots.append(root.cpu())
        tasks[str(c)] = trainer.tasks
    model = TreeClassifier(classes, roots)
    payload = {"spec": model.spec(), "model": cpu_state(model), "tasks": tasks,
               "features": bundle["fingerprint"], "manifest": bundle["manifest"],
               "config": config, "seconds_this_invocation": time.perf_counter() - start}
    save_checkpoint(Path(folder) / "tree.pt", payload)
    return model.to(device)


def load_tree(path, device):
    saved = load_checkpoint(path)
    model = TreeClassifier.from_spec(saved["spec"])
    model.load_state_dict(saved["model"])
    return model.to(device), saved


def feature_metrics(model, split, device, batch_size=1024):
    model.eval()
    errors, count, loss, confusion = 0, 0, 0., torch.zeros(model.classes, model.classes, dtype=torch.long)
    class_stats = [{"positive_correct": 0, "positive_count": 0, "negative_correct": 0, "negative_count": 0}
                   for _ in model.roots]
    before = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for xb, yb in zip(split["x"].split(batch_size), split["y"].split(batch_size)):
            x, y = xb.to(device), yb.to(device)
            prediction = model.predict(x)
            errors += int((prediction != y).sum())
            count += len(y)
            loss += float(model.loss(x, y)) * len(y)
            for actual, predicted in zip(y.cpu().tolist(), prediction.cpu().tolist()):
                confusion[actual, predicted] += 1
            for i, root in enumerate(model.roots):
                wanted = y == (1 if model.classes == 2 else i)
                got = root(x) >= 0
                stats = class_stats[i]
                stats["positive_correct"] += int((got & wanted).sum())
                stats["positive_count"] += int(wanted.sum())
                stats["negative_correct"] += int((~got & ~wanted).sum())
                stats["negative_count"] += int((~wanted).sum())
    if device.type == "cuda":
        torch.cuda.synchronize()
    for stats in class_stats:
        stats["positive_recall"] = stats["positive_correct"] / max(stats["positive_count"], 1)
        stats["negative_accuracy"] = stats["negative_correct"] / max(stats["negative_count"], 1)
    return {"examples": count, "errors": errors, "accuracy": 1 - errors / count,
            "loss": loss / count, "confusion": confusion.tolist(), "binary_trees": class_stats,
            "evaluation_seconds": time.perf_counter() - before}
