"""Build a standalone GitHub upload folder from reviewed, locked experiments."""
from pathlib import Path
import copy
import hashlib
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import torch

from .common import load_checkpoint, read_json, save_checkpoint, seed_everything, write_json
from .experiments import deployment_payload
from .review import collect_review
from .train import load_tree


def remove_inactive_positions(model):
    """Inactive predictions are always zero; deleting those subtrees preserves scores."""
    def trim(node):
        for side in ("A", "B"):
            child = getattr(node, side)
            if child is None:
                continue
            if not child.active:
                setattr(node, side, None)
            else:
                trim(child)
    for root in model.roots:
        if not root.active:
            raise ValueError("Cannot delete an inactive classifier root")
        trim(root)
    return model


def export_compact_models(runs_root, output_dir, device):
    review = collect_review(runs_root)
    seed_everything(409)
    records = {}
    for dataset, data in review["datasets"].items():
        run = Path(runs_root) / dataset
        bundle = load_checkpoint(run / "features.pt")
        name = data["smallest_near_perfect"]
        original, _ = load_tree(run / name / "tree.pt", device)
        compact = remove_inactive_positions(copy.deepcopy(original)).to(device).eval()
        original.eval()
        with torch.no_grad():
            for split in ("train", "validation"):
                for batch in bundle["splits"][split]["x"].split(512):
                    batch = batch.to(device)
                    if not torch.equal(original(batch), compact(batch)):
                        raise ValueError(f"Compaction changes classifier scores: {dataset}/{name}")
        path = Path(output_dir) / dataset / "compact-smallest-near-perfect.pt"
        payload = deployment_payload(compact, bundle)
        payload["source_experiment"] = name
        payload["compaction"] = "Removed only inactive correction positions; unchanged scores verified on full train and validation; no test reevaluation or retraining."
        save_checkpoint(path, payload)
        size = compact.size(bundle["trunk_parameters"])
        if size["allocated_parameters"] != size["active_parameters"]:
            raise ValueError("Inactive allocation remains in compact deployment")
        source = next(r for r in data["models"] if r["name"] == name)
        records[dataset] = {"source_experiment": name, "file": f"models/{dataset}/{path.name}", "local_file": str(path), "size": size,
                           "bytes": path.stat().st_size, "train": source["train"], "validation": source["validation"],
                           "test": source["test"], "test_metric_provenance": "Inherited from the unchanged, predeclared source model; no new test evaluation."}
    write_json(Path(output_dir).parent / "compact-models.json", records)
    return records


def build_upload_folder(runs_root="runs", output="github-upload/ELL409-neural-tree", device=torch.device("cpu")):
    collect_review(runs_root)
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output == root or root not in output.parents:
        raise ValueError("Upload destination must be a separate folder inside this project")
    output.mkdir(parents=True, exist_ok=True)
    # Do not delete a previous folder: copying is reversible, and the manifest
    # records every included file. Runtime data and hidden Git metadata are omitted.
    include = ["ell409_tree", "configs", "tests", "notebooks", "scripts", "docs", "results"]
    for directory in include:
        for source in sorted((root / directory).rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix not in (".pyc", ".pt"):
                destination = output / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    for name in ("README.md", "pyproject.toml", ".gitignore", "requirements-lock.txt", "implementation-status.json"):
        shutil.copy2(root / name, output / name)
    report = root / "output" / "ELL409_Neural_Tree_Report.pdf"
    (output / "report").mkdir(exist_ok=True)
    shutil.copy2(report, output / "report" / report.name)
    for dataset in ("cifar10", "cats_vs_dogs"):
        run = Path(runs_root) / dataset
        # Small provenance/evidence records and figures; no image dataset/cache.
        for source in run.rglob("*"):
            if source.is_file() and (source.suffix == ".json" or "figures" in source.parts and source.suffix == ".png"):
                dest = output / "evidence" / dataset / source.relative_to(run)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        for name in ("manifest.json", "duplicate-audit.json", "export_info.json"):
            source = root / "data" / dataset / name
            if source.exists():
                dest = output / "splits" / dataset / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        for name in ("adaptive-w64", "pruned"):
            dest = output / "models" / dataset / f"{name}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / name / "deployment.pt", dest)
    # Compact models were exported/verified locally before this publication step.
    for source in (root / "results" / "models").rglob("*.pt"):
        dest = output / "models" / source.relative_to(root / "results" / "models")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "UPLOAD_MANIFEST.json":
            manifest.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(output / "UPLOAD_MANIFEST.json", {"files": manifest, "total_bytes": sum(r["bytes"] for r in manifest),
               "repository_link": "Add the actual private GitHub URL after creating the repository; report content is complete.",
               "large_data_omitted": "Datasets, feature caches, optimizer journals, virtual environments, and Git metadata are not included."})
    archive = root / "dist" / "ELL409-GitHub-upload.zip"
    archive.parent.mkdir(exist_ok=True)
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file(): z.write(path, output.name + "/" + path.relative_to(output).as_posix())
    print(f"GitHub upload folder: {output}")
    print(f"Archive: {archive}; {len(manifest)} files; {sum(r['bytes'] for r in manifest):,} uncompressed bytes")
    return output
