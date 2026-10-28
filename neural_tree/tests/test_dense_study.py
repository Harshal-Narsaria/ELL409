import copy
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from ell409_tree import dense_study as dense
from ell409_tree.common import read_json, save_checkpoint, write_json


def test_reuse_requires_exact_optimization_and_feature_identity():
    config = {"width": 16, "node_epochs": 300, "seed": 409, "max_depth": 2}
    bundle = {"fingerprint": "features", "manifest": "split"}
    record = {"config": config, "features": "features", "manifest": "split"}
    assert dense.compatible(record, config, bundle)
    changed = copy.deepcopy(record)
    changed["config"]["node_epochs"] = 100
    assert not dense.compatible(changed, config, bundle)
    assert not dense.compatible(record, config, {**bundle, "fingerprint": "other-trunk"})
    assert not dense.compatible(record, config, {**bundle, "manifest": "other-split"})


def test_test_tensors_are_not_available_to_the_training_runner():
    class GuardedSplits(dict):
        def __getitem__(self, key):
            if key == "test": raise AssertionError("Test tensors must not be accessed")
            return super().__getitem__(key)
    splits = GuardedSplits(train=object(), validation=object(), test=object())
    full = {"splits": splits, "fingerprint": "features"}
    narrowed = dense.train_validation_bundle(full)
    assert set(narrowed["splits"]) == {"train", "validation"}
    assert narrowed["splits"]["train"] is splits["train"]
    assert "test" in full["splits"]


def test_resume_reuses_matching_runs_and_never_retrains_completed_widths(tmp_path, monkeypatch):
    study = {"datasets": ["toy"], "widths": [8, 12], "primary_seed": 409, "repeat_seeds": [],
             "repeat_widths": {"toy": []}, "node_epochs": 300, "correction_epochs": 150,
             "max_depth": 2, "max_nodes": 7, "source_prefix": "sweep-long", "policy": "validation only"}
    source, run, output = tmp_path / "source", tmp_path / "source/dense-study", tmp_path / "plots"
    bundle = {"fingerprint": "features", "manifest": "split", "trunk_hash": "trunk", "classes": 2,
              "splits": {s: {"x": torch.zeros(4, 2), "y": torch.tensor([0, 1, 0, 1])} for s in ("train", "validation", "test")}}
    base = {"seed": 409, "width": 64, "node_epochs": 100}
    save_checkpoint(source / "toy/features.pt", bundle)
    write_json(source / "toy/configuration.json", {"config": base})
    config_path = tmp_path / "config.json"; write_json(config_path, study)

    def record(width, config):
        errors = 1 if width == 8 else 0
        return {"name": f"source-w{width}", "config": config, "features": "features", "manifest": "split",
                "train": {"examples": 4, "errors": errors, "accuracy": 1-errors/4, "loss": .1},
                "validation": {"examples": 4, "errors": 1, "accuracy": .75, "loss": .5},
                "size": {"active_parameters": width * 10, "allocated_parameters": width * 70, "active_nodes": 1},
                "convergence": {"needs_extension": False}}
    write_json(source / "toy/sweep-long-w8-s409/metrics.json", record(8, dense.node_config(base, study, 8, 409)))
    wrong = dense.node_config(base, study, 12, 409); wrong["node_epochs"] = 100
    write_json(source / "toy/sweep-long-w12-s409/metrics.json", record(12, wrong))
    calls = []

    def fake_train(b, folder, name, config, device):
        assert "test" not in b["splits"]
        calls.append(config["width"])
        r = record(config["width"], config); r["name"] = name
        return r
    monkeypatch.setattr(dense, "run_tree", fake_train)
    monkeypatch.setattr(dense, "plot_curve", lambda *a, **kw: None)
    monkeypatch.setattr(dense, "package_extension", lambda *a: None)
    prepared = dense.dense_study(source, run, output, config_path, torch.device("cpu"), True)
    assert prepared["completed"] == 1 and not calls
    assert read_json(output / "interpretation.json")["toy"]["all_repeats_complete"] is False
    complete = dense.dense_study(source, run, output, config_path, torch.device("cpu"))
    assert complete["status"] == "complete" and calls == [12]
    dense.dense_study(source, run, output, config_path, torch.device("cpu"))
    assert calls == [12]
    assert "test" not in read_json(output / "measurements.json")["toy"][0]
    write_json(config_path, {**study, "node_epochs": 301})
    with pytest.raises(ValueError, match="identity changed"):
        dense.dense_study(source, run, output, config_path, torch.device("cpu"))


def test_completed_package_preserves_pdf_excludes_optimizer_journals_and_detects_edits(tmp_path):
    project, run, output = tmp_path / "project", tmp_path / "study", tmp_path / "results"
    source = project / "github-upload/ELL409-professor-view"
    for directory in ("configs", "scripts", "docs", "tests", "report", "ell409_tree", "results"):
        (source / directory).mkdir(parents=True, exist_ok=True)
    (source / "README.md").write_text("Original submission", encoding="utf-8")
    original_pdf = b"Original reviewed PDF"
    (source / "report/ELL409_Neural_Tree_Report.pdf").write_bytes(original_pdf)
    for relative in ("configs/dense-study.json", "scripts/run-dense-study.ps1", "docs/DENSE_STUDY.md", "tests/test_dense_study.py", "ell409_tree/dense_study.py"):
        p = project / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("content", encoding="utf-8")
    (project / "dist").mkdir()
    output.mkdir(); (output / "measurements.csv").write_text("width,error\n8,1\n", encoding="utf-8")
    (output / "README.md").write_text("Dense results", encoding="utf-8")
    write_json(run / "definition.json", {"uses_test_results": False})
    (run / "optimizer.pt").write_bytes(b"Local training state")
    target = Path(dense.package_extension(run, output, project))
    assert (target / "report/ELL409_Neural_Tree_Report.pdf").read_bytes() == original_pdf
    assert (source / "README.md").read_text(encoding="utf-8") == "Original submission"
    with ZipFile(output / "full-records.zip") as z:
        assert z.namelist() == ["definition.json"]
    with ZipFile(project / "dist/ELL409-professor-view-dense.zip") as z:
        assert z.testzip() is None
    assert dense.package_extension(run, output, project) == str(target)
    (target / "README.md").write_text("User edited this file", encoding="utf-8")
    with pytest.raises(ValueError, match="was edited"):
        dense.package_extension(run, output, project)
