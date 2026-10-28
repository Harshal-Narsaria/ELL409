from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from v2_tree.common import (cpu_state, digest, load_checkpoint, save_checkpoint,
                            seed_everything, study_lock, write_json)
from v2_tree.data import make_cache, normalized
from v2_tree.models import ImageForest, ImageNode, PixelScore, correction_targets
from v2_tree.study import fixed_definition, prune, study_main
from v2_tree.train import ImageTreeTrainer, epoch_budget, exact_signed_refit, fit_node, root_ablation


def test_reference_masks_and_signed_upward_corrections():
    p, y = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 1, 0])
    a, b = correction_targets(p, y).values()
    assert a[0].tolist() == [0, 0, 1, 0] and a[1].tolist() == [True, False, True, True]
    assert b[0].tolist() == [0, 0, 0, 1] and b[1].tolist() == [False, True, True, True]
    node = ImageNode(16, 4)
    node.A, node.B = ImageNode(16, 4), ImageNode(16, 4)
    with torch.no_grad():
        for model in (node.base, node.A.base, node.B.base):
            for parameter in model.parameters(): parameter.zero_()
        node.base.layers[-1].bias.fill_(-1)
        node.alpha.fill_(2); node.beta.zero_()
    x = torch.zeros(2, 3, 16, 16)
    assert node(x).tolist() == [1, 1]
    with torch.no_grad(): node.beta.fill_(3)
    assert node(x).tolist() == [-2, -2]


def test_exact_refit_corrects_both_error_types():
    inputs = torch.tensor([[-2., 0, 0], [2., 0, 0], [-2., 1, 0], [2., 0, 1]])
    y = torch.tensor([0., 1, 1, 0])
    weights = exact_signed_refit(inputs, y)
    assert weights is not None and (weights[:2] >= 0).all()
    final = inputs[:, 0] + weights[0]*inputs[:, 1] - weights[1]*inputs[:, 2] + weights[2]
    assert torch.equal((final >= 0).float(), y)


def test_hybrid_roundtrip_ablation_and_complete_parameter_count(tmp_path):
    seed_everything(42)
    root = ImageNode(16, 4)
    root.A = ImageNode(16, 4, "cnn", [4, 8])
    with torch.no_grad(): root.alpha.fill_(2)
    forest = ImageForest(2, [root])
    x = torch.randn(8, 3, 16, 16)
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"spec": forest.spec(), "model": cpu_state(forest)})
    saved = load_checkpoint(path)
    loaded = ImageForest.from_spec(saved["spec"])
    loaded.load_state_dict(saved["model"])
    assert torch.equal(forest(x), loaded(x))
    assert forest.size()["parameters"] == sum(p.numel() for p in forest.parameters())
    assert forest.size()["cnn_nodes"] == 1
    ablated = root_ablation(forest)
    assert ablated.size()["nodes"] == 1
    assert torch.equal(ablated(x), root.base(x))
    assert forest.roots[0].A is not None


def test_epoch_resume_matches_uninterrupted_optimization(tmp_path):
    seed_everything(8)
    x, y = torch.randn(12, 3, 8, 8), torch.tensor([0., 1]*6)
    indices = torch.arange(len(y))
    config = {"mlp_epochs": 4, "cnn_epochs": 4, "mlp_batch": 4, "cnn_batch": 4, "lr": .001, "eval_batch": 4}
    identity = {"path": "root", "kind": "mlp"}
    seed_everything(409); first = PixelScore(8, 4)
    uninterrupted = fit_node(first, x, indices, y, tmp_path / "full.pt", config, identity)
    seed_everything(409); second = PixelScore(8, 4)
    def interrupt(**fields):
        if fields["epoch"] == 2: raise RuntimeError("Simulated interruption")
    with pytest.raises(RuntimeError, match="Simulated"):
        fit_node(second, x, indices, y, tmp_path / "resume.pt", config, identity, interrupt)
    seed_everything(999); resumed = PixelScore(8, 4)
    result = fit_node(resumed, x, indices, y, tmp_path / "resume.pt", config, identity)
    assert [r["errors"] for r in result["history"]] == [r["errors"] for r in uninterrupted["history"]]
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), resumed.parameters()))


def test_pixel_cache_uses_train_statistics_and_never_opens_test(tmp_path):
    records = []
    for i, value in enumerate([20, 60, 100, 240]):
        image = np.full((8, 8, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(tmp_path / f"{i}.png")
        records.append({"id": str(i), "path": f"{i}.png", "label": i % 2})
    manifest = {"fingerprint": "fixture", "classes": ["cat", "dog"], "source": "fixture",
                "splits": {"train": records[:3], "validation": records[3:], "test": [{"id": "test", "path": "MUST_NOT_OPEN.png", "label": 0}]}}
    write_json(tmp_path / "manifest.json", manifest)
    cache = make_cache(tmp_path / "manifest.json", tmp_path / "pixels.pt", 8)
    assert set(cache["splits"]) == {"train", "validation"}
    assert torch.allclose(cache["mean"], torch.full((3,), 60/255))
    x = normalized(cache["splits"]["train"]["pixels"], cache, torch.device("cpu"))
    assert x.mean().abs() < 1e-6
    assert torch.allclose(x.std((0, 2, 3), unbiased=False), torch.ones(3), atol=1e-5)


def test_os_lock_and_frozen_protocol(tmp_path):
    path = tmp_path / "runner.lock"
    with study_lock(path):
        with pytest.raises(RuntimeError, match="already running"):
            with study_lock(path): pass
    with study_lock(path): pass
    fixed_definition(tmp_path / "definition.json", {"width": 16})
    with pytest.raises(ValueError, match="Frozen"):
        fixed_definition(tmp_path / "definition.json", {"width": 32})


def test_smoke_limit_samples_every_class_in_grouped_manifest(tmp_path):
    records = []
    for i in range(6):
        Image.fromarray(np.full((8, 8, 3), 20 + 20*i, dtype=np.uint8)).save(tmp_path / f"{i}.png")
        records.append({"id": str(i), "path": f"{i}.png", "label": int(i >= 3)})
    write_json(tmp_path / "manifest.json", {"fingerprint": "fixture", "source": "fixture", "classes": ["cat", "dog"],
        "splits": {"train": records, "validation": records, "test": []}})
    cache = make_cache(tmp_path / "manifest.json", tmp_path / "pixels.pt", 8, limit=2)
    assert cache["splits"]["train"]["y"].tolist() == [0, 1]
    assert cache["splits"]["train"]["ids"] == ["0", "3"]


def test_refuses_to_write_inside_original_project(tmp_path):
    original = tmp_path / "original"
    with pytest.raises(ValueError, match="outside"):
        study_main(original, original / "runs", tmp_path / "results", tmp_path / "config.json", torch.device("cpu"))


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_node_attribution_backpropagates_and_removes_hooks(kind):
    from v2_tree.visualize import attribution
    seed_everything(123)
    node = ImageNode(16, 4, kind, [4, 8])
    heat = attribution(node.base, torch.randn(1, 3, 16, 16), kind)
    assert heat.shape == (16, 16) and np.isfinite(heat).all()
    if kind == "cnn": assert not node.base.trunk[3]._forward_hooks


def test_pruning_tolerance_is_cumulative_not_per_removal(tmp_path):
    root = ImageNode(8, 4)
    root.A, root.B = ImageNode(8, 4), ImageNode(8, 4)
    with torch.no_grad():
        for node, coordinate in [(root, 0), (root.A, 1), (root.B, 2)]:
            for p in node.base.parameters(): p.zero_()
            node.base.layers[1].weight[0, coordinate] = 1
            node.base.layers[-1].weight[0, 0] = 2
            node.base.layers[-1].bias.fill_(-1)
        root.alpha.fill_(2); root.beta.fill_(2)
    model = ImageForest(2, [root])
    x = torch.zeros(10, 3, 8, 8)
    x[:, 0, 0, 0] = 1
    x[0, 0, 0, 2] = 1  # false positive, fixed by B
    x[1, 0, 0, 0] = 0; x[1, 0, 0, 1] = 1  # false negative, fixed by A
    y = torch.ones(10, dtype=torch.long); y[0] = 0
    assert torch.equal(model.predict(x), y)
    source = tmp_path / "source"
    save_checkpoint(source / "deployment.pt", {"spec": model.spec(), "model": cpu_state(model)})
    record = prune(tmp_path / "pruned", source, {"fingerprint": "fixture", "manifest": "fixture"},
                   x, y, x, y, {"eval_batch": 5, "pruning_tolerance": .1})
    assert record["validation"]["accuracy"] == .9
    assert sum(row["accepted"] for row in record["pruning"]) == 1
    assert record["minimum_allowed_validation_accuracy"] == .9


def test_transient_windows_replace_lock_is_retried(tmp_path, monkeypatch):
    import v2_tree.common as common
    actual_replace = common.os.replace
    calls = []
    def intermittently_locked(source, target):
        calls.append((source, target))
        if len(calls) < 3:
            raise PermissionError("Simulated Windows sharing violation")
        return actual_replace(source, target)
    monkeypatch.setattr(common.os, "replace", intermittently_locked)
    monkeypatch.setattr(common.time, "sleep", lambda seconds: None)
    common.write_json(tmp_path / "progress.json", {"epoch": 42})
    assert common.read_json(tmp_path / "progress.json") == {"epoch": 42}
    assert len(calls) == 3


def test_multiclass_uses_corrected_root_scores_for_argmax():
    roots = [ImageNode(8, 4) for _ in range(3)]
    with torch.no_grad():
        for i, root in enumerate(roots):
            for p in root.base.parameters(): p.zero_()
            root.base.layers[-1].bias.fill_(i - 2)
    model = ImageForest(3, roots)
    x = torch.zeros(2, 3, 8, 8)
    assert model.predict(x).tolist() == [2, 2]
    roots[1].A = ImageNode(8, 4)
    with torch.no_grad():
        for p in roots[1].A.base.parameters(): p.zero_()
        roots[1].A.base.layers[-1].bias.fill_(1)
        roots[1].alpha.fill_(3)
    assert model.predict(x).tolist() == [1, 1]
    wanted = torch.ones(2, dtype=torch.long)
    assert torch.equal(model.loss(x, wanted), torch.nn.functional.cross_entropy(model(x), wanted))


def test_fixed_root_budget_applies_only_to_mlp_roots(tmp_path):
    old = {"architecture": "hybrid", "width": 16, "cnn_depths": [1], "mlp_epochs": 240, "cnn_epochs": 90}
    images = torch.zeros(2, 3, 8, 8)
    legacy = ImageTreeTrainer(tmp_path, old, images, 409, "fixture")
    assert [legacy.width_at(d) for d in range(3)] == [16, 16, 16]
    fixed = {**old, "root_width": 4, "root_epochs": 120}
    trainer = ImageTreeTrainer(tmp_path, fixed, images, 409, "fixture")
    assert [trainer.width_at(d) for d in range(3)] == [4, 16, 16]
    assert epoch_budget(fixed, {"kind": "mlp", "depth": 0}) == 120
    assert epoch_budget(fixed, {"kind": "mlp", "depth": 2}) == 240
    assert epoch_budget(fixed, {"kind": "cnn", "depth": 1}) == 90
    assert epoch_budget(old, {"kind": "mlp", "depth": 0}) == 240
    cnn_control = ImageTreeTrainer(tmp_path, {**fixed, "architecture": "cnn_root"}, images, 409, "fixture")
    assert cnn_control.width_at(0) == 16
    assert epoch_budget(fixed, {"kind": "cnn", "depth": 0}) == 90


def test_fit_preserving_pruning_rejects_loss_of_training_fit(tmp_path):
    root = ImageNode(8, 4)
    root.A = ImageNode(8, 4)
    with torch.no_grad():
        for node in (root, root.A):
            for p in node.base.parameters(): p.zero_()
        root.base.layers[-1].bias.fill_(-1)
        root.A.base.layers[1].weight[0, 0] = 1
        root.A.base.layers[-1].weight[0, 0] = 2
        root.A.base.layers[-1].bias.fill_(-1)
        root.alpha.fill_(2)
    model = ImageForest(2, [root])
    x = torch.zeros(10, 3, 8, 8); x[0, 0, 0, 0] = 1
    y = torch.zeros(10, dtype=torch.long); y[0] = 1
    source = tmp_path / "source"
    save_checkpoint(source / "deployment.pt", {"spec": model.spec(), "model": cpu_state(model)})
    args = (source, {"fingerprint": "fixture", "manifest": "fixture"}, x, y, x, y)
    config = {"eval_batch": 5, "pruning_tolerance": .1, "near_perfect": .995}
    usual = prune(tmp_path / "validation-only", *args, config)
    protected = prune(tmp_path / "retain-fit", *args, {**config, "preserve_training_fit_when_pruning": True})
    assert usual["size"]["nodes"] == 1 and usual["train"]["accuracy"] == .9
    assert protected["size"]["nodes"] == 2 and protected["train"]["accuracy"] == 1.
    assert not protected["pruning"][0]["accepted"]
