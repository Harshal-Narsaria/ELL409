import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from ell409_tree.common import load_checkpoint, restore_rng, rng_state, save_checkpoint, seed_everything
from ell409_tree.data import grouped_split, synthetic_manifest
from ell409_tree.models import BaseNode, CorrectionNode, TreeClassifier, correction_targets
from ell409_tree.train import fit_binary, load_tree, cpu_state


def test_reference_table_and_dont_care_masks():
    p = torch.tensor([0, 1, 0, 1])
    y = torch.tensor([0, 1, 1, 0])
    targets = correction_targets(p, y)
    assert targets["A"][0].tolist() == [0, 0, 1, 0]
    assert targets["A"][1].tolist() == [True, False, True, True]
    assert targets["B"][0].tolist() == [0, 0, 0, 1]
    assert targets["B"][1].tolist() == [False, True, True, True]


def set_constant(node, score):
    with torch.no_grad():
        for p in node.base.parameters():
            p.zero_()
        node.base.layers[-1].bias.fill_(score)


def test_signed_upward_corrections_and_hard_recursive_decisions():
    x = torch.ones(2, 3)
    root = CorrectionNode(3, 4)
    set_constant(root, -2.4)
    root.A = CorrectionNode(3, 4)
    set_constant(root.A, 1)
    with torch.no_grad():
        root.alpha.fill_(3.1)
    assert torch.allclose(root(x), torch.full((2,), .7), atol=1e-6)
    root.B = CorrectionNode(3, 4)
    set_constant(root.B, -1)
    root.B.A = CorrectionNode(3, 4)
    set_constant(root.B.A, 1)
    with torch.no_grad():
        root.B.alpha.fill_(2)
        root.beta.fill_(2)
    assert torch.allclose(root(x), torch.full((2,), -1.3), atol=1e-6)
    # A negative magnitude is clamped; it cannot reverse a correction's sign.
    with torch.no_grad():
        root.beta.fill_(-4)
    assert torch.allclose(root(x), torch.full((2,), .7), atol=1e-6)


def test_recursive_checkpoint_round_trip_and_allocated_size(tmp_path):
    seed_everything(12)
    root = CorrectionNode(3, 5)
    root.A = CorrectionNode(3, 5)
    root.A.B = CorrectionNode(3, 5)
    root.pad(2)
    model = TreeClassifier(2, [root])
    x = torch.randn(11, 3)
    expected = model(x)
    path = tmp_path / "tree.pt"
    save_checkpoint(path, {"spec": model.spec(), "model": cpu_state(model)})
    actual, _ = load_tree(path, torch.device("cpu"))
    assert torch.equal(actual(x), expected)
    assert actual.size(100)["allocated_nodes"] == 7
    assert actual.size(100)["active_nodes"] == 3
    assert actual.size(100)["allocated_parameters"] == sum(p.numel() for p in model.parameters()) + 100
    assert actual.size(100)["active_parameters"] < actual.size(100)["allocated_parameters"]


def test_optimizer_and_rng_resume_matches_uninterrupted_training(tmp_path):
    config = {"node_lr": .01, "node_epochs": 6, "feature_batch_size": 9}
    seed_everything(77)
    x = torch.randn(30, 3)
    y = (x[:, 0] * x[:, 1] > 0).float()
    initial = BaseNode(3, 7)
    seed_everything(500)
    full = copy.deepcopy(initial)
    full_stats = fit_binary(full, x, y, tmp_path / "full.pt", config, "same-task")
    seed_everything(500)
    part = copy.deepcopy(initial)
    fit_binary(part, x, y, tmp_path / "part.pt", {**config, "node_epochs": 2}, "same-task")
    seed_everything(991)  # Simulate unrelated initialization in a fresh process.
    resumed = BaseNode(3, 7)
    resumed_stats = fit_binary(resumed, x, y, tmp_path / "part.pt", config, "same-task")
    for a, b in zip(full.parameters(), resumed.parameters()):
        assert torch.equal(a, b)
    assert full_stats["history"] == resumed_stats["history"]
    checkpoint = load_checkpoint(tmp_path / "part.pt")
    assert checkpoint["epoch"] == 6
    assert checkpoint["optimizer"]["state"]
    assert set(checkpoint["rng"]) == {"python", "numpy", "torch", "cuda"}


def test_duplicate_groups_do_not_cross_splits_and_labels_are_checked():
    records = [{"id": str(i), "label": i % 2, "hash": str(i)} for i in range(40)]
    records += [{"id": "copy", "label": 0, "hash": "0"}]
    splits = grouped_split(records, {"train": .8, "validation": .1, "test": .1})
    owners = [name for name, rs in splits.items() if any(r["hash"] == "0" for r in rs)]
    assert len(owners) == 1
    assert sum(len(rs) for rs in splits.values()) == len(records)
    broken = records + [{"id": "conflict", "label": 1, "hash": "0"}]
    with pytest.raises(ValueError, match="conflicting"):
        grouped_split(broken, {"train": .8, "test": .2})


def test_resume_rejects_changed_training_examples(tmp_path):
    config = {"node_lr": .01, "node_epochs": 1, "feature_batch_size": 4}
    x, y = torch.ones(8, 3), torch.zeros(8)
    fit_binary(BaseNode(3, 4), x, y, tmp_path / "node.pt", config, "task")
    with pytest.raises(ValueError, match="changed"):
        fit_binary(BaseNode(3, 4), x * 2, y, tmp_path / "node.pt", config, "task")


def test_multiclass_reports_final_argmax_not_binary_average():
    roots = [CorrectionNode(3, 4) for _ in range(3)]
    for i, root in enumerate(roots):
        set_constant(root, i)
    model = TreeClassifier(3, roots)
    assert model.predict(torch.ones(5, 3)).tolist() == [2] * 5


def test_pruning_tolerance_is_cumulative(tmp_path, monkeypatch):
    from ell409_tree import experiments
    root = CorrectionNode(3, 4)
    root.A = CorrectionNode(3, 4)
    root.B = CorrectionNode(3, 4)
    model = TreeClassifier(2, [root])
    source = tmp_path / "source"
    source.mkdir()
    save_checkpoint(source / "tree.pt", {"spec": model.spec(), "model": cpu_state(model), "tasks": {}, "config": {}})
    def metrics(m, split, device):
        removed = 3 - m.size()["active_nodes"]
        return {"accuracy": .99 - .004 * removed, "errors": removed}
    monkeypatch.setattr(experiments, "feature_metrics", metrics)
    monkeypatch.setattr(experiments, "inference_benchmark", lambda *a: {})
    bundle = {"splits": {"train": {}, "validation": {}}, "trunk_parameters": 10,
              "fingerprint": "features", "manifest": "data", "feature_mean": torch.zeros(3),
              "feature_std": torch.ones(3), "trunk_hash": "trunk"}
    result = experiments.prune_tree(bundle, tmp_path, "source", torch.device("cpu"))
    assert result["validation"]["accuracy"] >= .985
    assert sum(row["accepted"] for row in result["pruning"]) == 1


def test_image_level_gradient_maps_are_finite():
    from ell409_tree.models import SharedCNN
    from ell409_tree.visualize import grad_cam, occlusion_map
    cnn, node = SharedCNN(2).eval(), CorrectionNode(128, 4)
    for p in cnn.parameters():
        p.requires_grad_(False)
    image = torch.randn(3, 32, 32)
    mean, std = torch.zeros(128), torch.ones(128)
    cam, score = grad_cam(cnn, node, image, mean, std)
    occ = occlusion_map(cnn, node, image, mean, std)
    assert cam.shape == occ.shape == (32, 32)
    assert np.isfinite(cam).all() and np.isfinite(occ).all() and np.isfinite(score)


def test_actual_recursive_growth_corrects_forced_parent_errors(tmp_path, monkeypatch):
    from ell409_tree import train
    original = train.fit_binary
    def fit(model, x, y, checkpoint, config, identity, projection=None):
        result = original(model, x, y, checkpoint, config, identity, projection)
        if isinstance(identity, dict) and identity.get("path") == "root" and identity.get("stage") != "correction":
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()  # Root predicts 1 for everyone; B must correct the negatives.
        return result
    monkeypatch.setattr(train, "fit_binary", fit)
    seed_everything(9)
    x = torch.tensor([[-1., 0.]] * 12 + [[1., 0.]] * 12)
    y = torch.tensor([0.] * 12 + [1.] * 12)
    config = {"width": 8, "node_epochs": 30, "node_lr": .03, "feature_batch_size": 24,
              "max_depth": 1, "max_nodes": 3, "correction_epochs": 80}
    trainer = train.TreeTrainer(tmp_path, config, 2, 9, torch.device("cpu"))
    node = trainer.grow(x, y, torch.arange(len(y)))
    assert node.B is not None and node.A is None
    assert torch.equal((node(x) >= 0).float(), y)
    assert trainer.tasks["root.B"]["labels"].tolist() == [1.] * 12 + [0.] * 12
    # A second invocation must recover the same completed recursive task.
    again = train.TreeTrainer(tmp_path, config, 2, 9, torch.device("cpu")).grow(x, y, torch.arange(len(y)))
    assert torch.equal(node(x), again(x))


def test_held_out_candidate_lock_rejects_new_experiments(tmp_path, monkeypatch):
    from ell409_tree import experiments
    from ell409_tree.common import write_json, digest
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"fingerprint": "data"})
    write_json(tmp_path / "selected-tree.json", {"name": "chosen"})
    write_json(tmp_path / "sweep.json", [])
    write_json(tmp_path / "pruned" / "metrics.json", {})
    write_json(tmp_path / "new" / "metrics.json", {})
    write_json(tmp_path / "test-declaration.json", {"manifest": "data", "candidates": ["pruned"], "configuration": digest({})})
    with pytest.raises(ValueError, match="declared already"):
        experiments.finalize_tests(manifest, tmp_path, {}, torch.device("cpu"))


def test_complete_deployment_matches_cached_feature_scores():
    from ell409_tree.models import SharedCNN, ImageTreeClassifier
    cnn = SharedCNN(2).eval()
    tree = TreeClassifier(2, [CorrectionNode(128, 4)]).eval()
    bundle = {"trunk_state": cpu_state(cnn.blocks), "image_mean": [.2, .3, .4], "image_std": [.1, .2, .3],
              "feature_mean": torch.randn(128), "feature_std": torch.rand(128) + .1}
    deployed = ImageTreeClassifier(copy.deepcopy(tree), bundle).eval()
    image = torch.rand(3, 3, 32, 32)
    normalized = (image - torch.tensor(bundle["image_mean"])[None, :, None, None]) / torch.tensor(bundle["image_std"])[None, :, None, None]
    expected = tree((cnn.features(normalized) - bundle["feature_mean"]) / bundle["feature_std"])
    assert torch.allclose(deployed(image), expected, atol=1e-6)


def test_windows_archive_normalization_preserves_official_image_names():
    from ell409_tree.data import archive_name
    assert archive_name(r"PetImages\Cat\0.jpg") == "PetImages/Cat/0.jpg"
    assert archive_name("PetImages/Dog/14.jpg") == "PetImages/Dog/14.jpg"


def test_exact_parent_refit_handles_large_base_errors_and_reports_infeasibility():
    from ell409_tree.train import exact_correction_candidate, CorrectionFit
    x = torch.tensor([[-80., 1., 0.], [90., 0., 1.], [-2., 0., 0.], [3., 0., 0.]])
    y = torch.tensor([1., 0., 0., 1.])
    weights = exact_correction_candidate(x, y)
    assert weights is not None and (weights[:2] >= 0).all()
    score = x[:, 0] + weights[0] * x[:, 1] - weights[1] * x[:, 2] + weights[2]
    assert ((2 * y - 1) * score >= .9999).all()
    impossible = torch.tensor([[0., 0., 0.], [0., 0., 0.]])
    assert exact_correction_candidate(impossible, torch.tensor([0., 1.])) is None
    # Fixed base scores already classify correctly, but cannot both reach margin 1
    # through a common offset when neither child fires. Infeasible != unclassifiable.
    small_margin = torch.tensor([[-.01, 0., 0.], [.01, 0., 0.]])
    assert exact_correction_candidate(small_margin, torch.tensor([0., 1.])) is None


def test_normalization_uses_only_training_pixels(tmp_path):
    from ell409_tree.common import read_json
    manifest_path = synthetic_manifest(tmp_path)
    manifest = read_json(manifest_path)
    images = np.load(manifest_path.parent / "images.npy").astype(np.float64) / 255
    indices = [r["index"] for r in manifest["splits"]["train"]]
    pixels = images[indices].reshape(-1, 3)
    assert np.allclose(manifest["normalization"]["mean"], pixels.mean(0), atol=1e-12)
    assert np.allclose(manifest["normalization"]["std"], pixels.std(0), atol=1e-12)
    assert manifest["normalization"]["computed_on"] == "train"


def test_lossless_resized_export_matches_training_resize(tmp_path):
    from PIL import Image
    original = Image.fromarray(np.random.default_rng(2).integers(0, 256, (211, 317, 3), dtype=np.uint8))
    expected = np.array(original.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR))
    path = tmp_path / "image.png"
    original.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR).save(path)
    with Image.open(path) as saved:
        actual = np.array(saved.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR))
    assert np.array_equal(actual, expected)


def test_double_descent_summary_distinguishes_missing_regime_from_candidate():
    from ell409_tree.experiments import interpret_sweep
    def records(errors, validation):
        return [{"config": {"width": 2 ** (i + 1)}, "train": {"errors": e}, "validation": {"accuracy": 1 - v}}
                for i, (e, v) in enumerate(zip(errors, validation))]
    no_threshold = records([10, 5, 2], [.4, .3, .2])
    assert "not established" in interpret_sweep(no_threshold, no_threshold)["observation"]
    already_fits = records([0, 0, 0], [.3, .2, .1])
    assert "already interpolates" in interpret_sweep(already_fits, already_fits)["observation"]
    candidate = records([10, 5, 0, 0, 0], [.5, .2, .35, .25, .15])
    assert interpret_sweep(candidate, candidate)["candidate_double_descent"]


def test_plot_regeneration_ignores_generated_interpretation_files(tmp_path):
    from ell409_tree.common import write_json
    from ell409_tree.experiments import plot_results
    records = [{"config": {"width": width}, "size": {"allocated_parameters": width * 10},
                "train": {"errors": 0, "accuracy": 1., "loss": .1},
                "validation": {"accuracy": .8, "loss": .4}} for width in (4, 8)]
    write_json(tmp_path / "sweep.json", records)
    plot_results(tmp_path)
    plot_results(tmp_path)
    assert (tmp_path / "sweep-interpretation.json").exists()
    assert (tmp_path / "figures" / "sweep.png").exists()


def test_conflict_exclusion_removes_entire_group_without_relabeling():
    from ell409_tree.data import exclude_conflicting_duplicates
    records = [{"id": "a", "label": 0, "hash": "same"}, {"id": "b", "label": 1, "hash": "same"},
               {"id": "c", "label": 0, "hash": "different"}]
    kept, groups = exclude_conflicting_duplicates(records)
    assert kept == records[2:]
    assert groups == [records[:2]]
    assert records[0]["label"] == 0 and records[1]["label"] == 1


def test_validation_epoch_history_resumes_and_never_selects_weights(tmp_path):
    seed_everything(25)
    x, vx = torch.randn(20, 3), torch.randn(8, 3)
    y, vy = (x[:, 0] > 0).float(), (vx[:, 0] > 0).float()
    initial = BaseNode(3, 4)
    config = {"node_lr": .01, "node_epochs": 4, "feature_batch_size": 5}
    seed_everything(40)
    full = copy.deepcopy(initial)
    expected = fit_binary(full, x, y, tmp_path / "full.pt", config, "epoch", validation=(vx, vy))
    seed_everything(40)
    part = copy.deepcopy(initial)
    fit_binary(part, x, y, tmp_path / "part.pt", {**config, "node_epochs": 2}, "epoch", validation=(vx, vy))
    actual = fit_binary(BaseNode(3, 4), x, y, tmp_path / "part.pt", config, "epoch", validation=(vx, vy))
    assert actual["history"] == expected["history"]
    assert len(actual["history"]) == 4
    # Reversing validation labels must not change the chosen training weights.
    seed_everything(40)
    reversed_validation = copy.deepcopy(initial)
    fit_binary(reversed_validation, x, y, tmp_path / "other.pt", config, "epoch", validation=(vx, 1 - vy))
    assert all(torch.equal(a, b) for a, b in zip(full.parameters(), reversed_validation.parameters()))
    with pytest.raises(ValueError, match="changed"):
        fit_binary(BaseNode(3, 4), x, y, tmp_path / "part.pt", config, "epoch", validation=(vx, 1 - vy))


def test_removing_inactive_positions_preserves_recursive_scores_and_removes_stored_parameters():
    from ell409_tree.submission import remove_inactive_positions
    seed_everything(333)
    root = CorrectionNode(3, 4)
    root.A = CorrectionNode(3, 4)
    with torch.no_grad(): root.alpha.fill_(2.)
    root.pad(3)
    model = TreeClassifier(2, [root])
    x = torch.randn(37, 3)
    original = model(x)
    allocated = model.size(100)["allocated_parameters"]
    compact = remove_inactive_positions(copy.deepcopy(model))
    assert torch.equal(compact(x), original)
    size = compact.size(100)
    assert size["allocated_parameters"] == size["active_parameters"] < allocated
    assert size["active_nodes"] == 2


def test_submission_audit_detects_incomplete_or_changed_locked_evaluation(tmp_path):
    from ell409_tree.common import digest, write_json
    from ell409_tree.review import verify_test_records
    config = {"seed": 409}
    audit = {"fingerprint": "source", "counts": {"train": 10, "validation": 5, "test": 7}}
    declaration = {"manifest": "source", "candidates": ["model"], "configuration": digest(config)}
    write_json(tmp_path / "test-declaration.json", declaration)
    def metric(n): return {"examples": n, "errors": 1, "accuracy": 1 - 1 / n}
    write_json(tmp_path / "model" / "metrics.json", {"manifest": "source", "train": metric(10), "validation": metric(5)})
    write_json(tmp_path / "model" / "test-metrics.json", {"declaration": digest(declaration), **metric(7)})
    assert verify_test_records(tmp_path, audit, config) == ["model"]
    write_json(tmp_path / "model" / "test-metrics.json", {"declaration": digest(declaration), **metric(6)})
    with pytest.raises(ValueError, match="Incomplete"):
        verify_test_records(tmp_path, audit, config)
    write_json(tmp_path / "new" / "metrics.json", {})
    with pytest.raises(ValueError, match="locked"):
        verify_test_records(tmp_path, audit, config)
