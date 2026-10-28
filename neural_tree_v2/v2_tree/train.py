from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import (cpu_state, digest, load_checkpoint, restore_rng, rng_state,
                     save_checkpoint, seed_everything, tensor_hash)
from .models import ImageForest, ImageNode, correction_targets


def scores_on(model, images, indices, batch=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(images[i]) for i in indices.split(batch)])


def score_metrics(scores, y):
    return int(((scores.detach() >= 0) != y.bool()).sum()), float(F.binary_cross_entropy_with_logits(scores.detach(), y.float()))


def epoch_budget(config, identity):
    if identity["kind"] == "cnn":
        return config["cnn_epochs"]
    if identity.get("depth", 0) == 0:
        return config.get("root_epochs", config["mlp_epochs"])
    return config["mlp_epochs"]


def fit_node(model, images, indices, y, path, config, identity, progress=None):
    """Index shared images rather than copying image tensors at each recursion."""
    epochs = epoch_budget(config, identity)
    batch = config["cnn_batch"] if identity["kind"] == "cnn" else config["mlp_batch"]
    signature = digest({"identity": identity, "config": config, "task": tensor_hash(indices, y)})
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0)
    start, history, elapsed = 0, [], 0.
    best_key = score_metrics(scores_on(model, images, indices, config["eval_batch"]), y)
    best_model = cpu_state(model)
    if Path(path).exists():
        saved = load_checkpoint(path)
        if saved["signature"] != signature:
            raise ValueError("Resume identity changed; use a new model name/directory.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, history, elapsed = saved["epoch"], saved["history"], saved["elapsed"]
        best_key, best_model = tuple(saved["best_key"]), saved["best_model"]
    positives = float(y.sum())
    weight = torch.tensor((len(y) - positives) / positives if 0 < positives < len(y) else 1., device=y.device)
    for epoch in range(start, epochs):
        before = time.perf_counter()
        model.train()
        order = torch.randperm(len(y), device=y.device)
        for positions in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images[indices[positions]])
            loss = F.binary_cross_entropy_with_logits(logits, y[positions], pos_weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite node loss")
            loss.backward()
            optimizer.step()
        if images.is_cuda:
            torch.cuda.synchronize()
        errors, loss = score_metrics(scores_on(model, images, indices, config["eval_batch"]), y)
        seconds = time.perf_counter() - before
        elapsed += seconds
        history.append({"epoch": epoch + 1, "errors": errors, "examples": len(y), "loss": loss, "seconds": seconds})
        if (errors, loss) < best_key:
            best_key, best_model = (errors, loss), cpu_state(model)
        save_checkpoint(path, {"signature": signature, "epoch": epoch + 1, "model": cpu_state(model),
                              "optimizer": optimizer.state_dict(), "rng": rng_state(), "history": history,
                              "best_key": best_key, "best_model": best_model, "elapsed": elapsed})
        if progress:
            progress(node=identity["path"], node_kind=identity["kind"], epoch=epoch + 1,
                     epochs=epochs, node_examples=len(y), node_errors=errors, epoch_seconds=seconds)
        if epoch == start or (epoch + 1) % 20 == 0 or epoch + 1 == epochs:
            print(f"{identity['path']} ({identity['kind']}) epoch {epoch+1}/{epochs}: {errors}/{len(y)} errors, {seconds:.2f}s", flush=True)
    model.load_state_dict(best_model)
    return {"history": history, "best_errors": best_key[0], "best_loss": best_key[1], "elapsed": elapsed}


def exact_signed_refit(inputs, labels):
    """Margin-one feasibility for fixed scores and binary child outputs."""
    import numpy as np
    from scipy.optimize import linprog

    values, targets = inputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
    rows, bounds = [], []
    for label in (0, 1):
        sign = 2 * label - 1
        for a in (0, 1):
            for b in (0, 1):
                mask = (targets == label) & (values[:, 1] == a) & (values[:, 2] == b)
                if mask.any():
                    rows.append(-sign * np.array([a, -b, 1.]))
                    bounds.append(float(np.min(sign * values[mask, 0])) - 1.)
    result = linprog([1., 1., 0.], A_ub=rows, b_ub=bounds,
                     bounds=[(0, None), (0, None), (None, None)], method="highs")
    if not result.success:
        return None
    return torch.tensor(result.x, dtype=inputs.dtype, device=inputs.device)


def fit_correction(inputs, y, path, epochs, identity):
    signature = digest({"identity": identity, "task": tensor_hash(inputs, y), "epochs": epochs})
    weights = nn.Parameter(torch.zeros(3, device=inputs.device))
    optimizer = torch.optim.Adam([weights], lr=.05)
    def logits():
        return inputs[:, 0] + weights[0] * inputs[:, 1] - weights[1] * inputs[:, 2] + weights[2]
    best_key, best_weights = score_metrics(logits(), y), weights.detach().clone()
    start, history, elapsed = 0, [], 0.
    if Path(path).exists():
        saved = load_checkpoint(path)
        if saved["signature"] != signature:
            raise ValueError("Correction resume identity changed")
        with torch.no_grad():
            weights.copy_(saved["weights"].to(inputs.device))
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, history, elapsed = saved["epoch"], saved["history"], saved["elapsed"]
        best_key, best_weights = tuple(saved["best_key"]), saved["best_weights"].to(inputs.device)
    for epoch in range(start, epochs):
        before = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(logits(), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite correction loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights[:2].clamp_(min=0)
        key = score_metrics(logits(), y)
        if key < best_key:
            best_key, best_weights = key, weights.detach().clone()
        elapsed += time.perf_counter() - before
        history.append({"epoch": epoch + 1, "errors": key[0], "loss": key[1]})
        save_checkpoint(path, {"signature": signature, "weights": weights.detach().cpu(),
                              "optimizer": optimizer.state_dict(), "rng": rng_state(), "epoch": epoch + 1,
                              "history": history, "elapsed": elapsed, "best_key": best_key,
                              "best_weights": best_weights.cpu()})
    exact = exact_signed_refit(inputs, y)
    if exact is not None:
        candidate = inputs[:, 0] + exact[0] * inputs[:, 1] - exact[1] * inputs[:, 2] + exact[2]
        if score_metrics(candidate, y) < best_key:
            best_weights = exact
    return best_weights, {"elapsed": elapsed, "history": history, "margin_one_feasible": exact is not None}


class ImageTreeTrainer:
    def __init__(self, folder, config, images, seed, fingerprint, progress=None):
        self.folder, self.config, self.images = Path(folder), config, images
        self.seed, self.fingerprint, self.progress = seed, fingerprint, progress
        self.used, self.tasks = 0, {}

    def kind_at(self, depth):
        if self.config["architecture"] == "cnn_root":
            return "cnn"
        return "cnn" if self.config["architecture"] == "hybrid" and depth in self.config["cnn_depths"] else "mlp"

    def width_at(self, depth):
        return self.config.get("root_width", self.config["width"]) if depth == 0 and self.kind_at(depth) == "mlp" else self.config["width"]

    def grow(self, indices, y, path="root", depth=0):
        self.used += 1
        kind = self.kind_at(depth)
        seed_everything(self.seed + int(digest(path)[:7], 16))
        node = ImageNode(self.config["resolution"], self.width_at(depth), kind, self.config["channels"]).to(self.images.device)
        identity = {"path": path, "depth": depth, "kind": kind, "seed": self.seed, "pixels": self.fingerprint}
        signature = digest({"identity": identity, "config": self.config, "task": tensor_hash(indices, y)})
        final = self.folder / (path + "-final.pt")
        if final.exists():
            saved = load_checkpoint(final)
            if saved["signature"] != signature:
                raise ValueError("Completed node identity changed")
            node = ImageNode.from_spec(saved["spec"]).to(self.images.device)
            node.load_state_dict(saved["model"])
            self.used += sum(1 for _ in node.walk()) - 1
            self.tasks.update(saved["tasks"])
            return node
        stats = fit_node(node.base, self.images, indices, y, self.folder / (path + "-base.pt"),
                         self.config, identity, self.progress)
        base_scores = scores_on(node.base, self.images, indices, self.config["eval_batch"])
        predictions = base_scores >= 0
        before = score_metrics(base_scores, y)
        self.tasks[path] = {"indices": indices.cpu(), "labels": y.cpu(), "base_predictions": predictions.cpu(),
                            "kind": kind, "base": stats, "errors_before": before[0]}
        targets = correction_targets(predictions, y)
        if before[0] and depth < self.config["max_depth"]:
            for side in ("A", "B"):
                child_y, mask = targets[side]
                if self.used < self.config["max_nodes"] and child_y[mask].sum() > 0:
                    setattr(node, side, self.grow(indices[mask], child_y[mask], path + "." + side, depth + 1))
        if node.A is not None or node.B is not None:
            a = (scores_on(node.A, self.images, indices, self.config["eval_batch"]) >= 0).float() if node.A is not None else torch.zeros_like(y)
            b = (scores_on(node.B, self.images, indices, self.config["eval_batch"]) >= 0).float() if node.B is not None else torch.zeros_like(y)
            inputs = torch.stack([base_scores, a, b], dim=1)
            weights, correction = fit_correction(inputs, y, self.folder / (path + "-correction.pt"),
                                                 self.config["correction_epochs"], identity)
            with torch.no_grad():
                node.alpha.copy_(weights[0]); node.beta.copy_(weights[1]); node.offset.copy_(weights[2])
            after = score_metrics(scores_on(node, self.images, indices, self.config["eval_batch"]), y)
            if after > before:
                with torch.no_grad():
                    node.alpha.zero_(); node.beta.zero_(); node.offset.zero_()
            self.tasks[path]["correction"] = correction
        after = score_metrics(scores_on(node, self.images, indices, self.config["eval_batch"]), y)
        self.tasks[path]["errors_after"] = after[0]
        own_tasks = {p: t for p, t in self.tasks.items() if p == path or p.startswith(path + ".")}
        save_checkpoint(final, {"signature": signature, "spec": node.spec(), "model": cpu_state(node), "tasks": own_tasks})
        print(f"{path} finished: errors {before[0]} -> {after[0]}, {kind}, depth {depth}", flush=True)
        return node


def train_forest(images, labels, folder, config, fingerprint, progress=None):
    classes = config["classes"]
    roots, tasks = [], {}
    for c in ([1] if classes == 2 else range(classes)):
        def node_progress(**fields):
            if progress:
                progress(class_index=c, **fields)
        trainer = ImageTreeTrainer(Path(folder) / f"class-{c}", config, images, config["seed"] + c, fingerprint, node_progress)
        root = trainer.grow(torch.arange(len(labels), device=images.device), (labels == c).float())
        roots.append(root.cpu())
        tasks[str(c)] = trainer.tasks
    return ImageForest(classes, roots).to(images.device), tasks


def root_ablation(model):
    """Same learned root weights, with every child and correction removed."""
    result = copy.deepcopy(model)
    for root in result.roots:
        root.A = root.B = None
        with torch.no_grad():
            root.alpha.zero_(); root.beta.zero_(); root.offset.zero_()
    return result


def evaluate(model, images, labels, batch=256):
    model.eval()
    errors, loss, count = 0, 0., 0
    binary = [{"positive_correct": 0, "positive_count": 0, "negative_correct": 0, "negative_count": 0} for _ in model.roots]
    confusion = torch.zeros(model.classes, model.classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in zip(images.split(batch), labels.split(batch)):
            scores = model(x)
            pred = (scores >= 0).long() if model.classes == 2 else scores.argmax(1)
            errors += int((pred != y).sum())
            loss += float(F.binary_cross_entropy_with_logits(scores, y.float(), reduction="sum") if model.classes == 2 else F.cross_entropy(scores, y, reduction="sum"))
            count += len(y)
            confusion += torch.bincount((model.classes * y + pred).cpu(), minlength=model.classes**2).reshape(model.classes, model.classes)
            for i, root in enumerate(model.roots):
                wanted = y == (1 if model.classes == 2 else i)
                got = scores >= 0 if model.classes == 2 else scores[:, i] >= 0
                b = binary[i]
                b["positive_correct"] += int((got & wanted).sum()); b["positive_count"] += int(wanted.sum())
                b["negative_correct"] += int((~got & ~wanted).sum()); b["negative_count"] += int((~wanted).sum())
    for b in binary:
        b["positive_recall"] = b["positive_correct"] / max(b["positive_count"], 1)
        b["negative_accuracy"] = b["negative_correct"] / max(b["negative_count"], 1)
    return {"examples": count, "errors": errors, "accuracy": 1 - errors / count, "loss": loss / count,
            "confusion": confusion.tolist(), "binary_trees": binary}
