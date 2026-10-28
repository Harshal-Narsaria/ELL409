from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import digest, read_json, write_json

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def image_hash(image):
    array = np.asarray(image, dtype=np.uint8)
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def grouped_split(records, fractions, seed=409):
    """Keep identical decoded images together; reject contradictory labels."""
    groups = defaultdict(list)
    for r in records:
        groups[r.get("input_hash", r["hash"])].append(r)
    by_class = defaultdict(list)
    for group in groups.values():
        labels = {r["label"] for r in group}
        if len(labels) != 1:
            raise ValueError("An exact image duplicate has conflicting labels: " + str([r["id"] for r in group]))
        by_class[group[0]["label"]].append(group)
    rng = np.random.default_rng(seed)
    result = {name: [] for name in fractions}
    for label in sorted(by_class):
        class_groups = sorted(by_class[label], key=lambda g: g[0]["hash"])
        rng.shuffle(class_groups)
        total = sum(len(g) for g in class_groups)
        counts = {name: 0 for name in fractions}
        for group in class_groups:
            # Largest remaining deficit preserves class proportions to group-size precision.
            name = max(fractions, key=lambda n: fractions[n] * total - counts[n])
            result[name].extend(group)
            counts[name] += len(group)
    return result


def prepare_cifar(data_dir, seed=409, download=True):
    from torchvision.datasets import CIFAR10
    folder = Path(data_dir) / "cifar10"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        old = read_json(folder / "manifest.json")
        if old["seed"] != seed:
            raise ValueError("Existing split seed differs. Use a separate data directory.")
        return folder / "manifest.json"
    train = CIFAR10(str(folder / "raw"), train=True, download=download)
    test = CIFAR10(str(folder / "raw"), train=False, download=download)
    images = np.concatenate([train.data, test.data])
    np.save(folder / "images.npy", images)
    records = [{"id": f"cifar-{i:05d}", "index": i, "label": int(y), "hash": image_hash(image)}
               for i, (image, y) in enumerate(zip(images, train.targets + test.targets))]
    official_train, official_test = records[:len(train)], records[len(train):]
    # Official test ownership wins: exclude matching training copies, never leak them.
    test_hashes = {r["hash"] for r in official_test}
    all_labels = defaultdict(set)
    for r in records:
        all_labels[r["hash"]].add(r["label"])
    if any(len(labels) > 1 for labels in all_labels.values()):
        raise ValueError("CIFAR contains conflicting labels on exact decoded duplicates; audit before proceeding.")
    overlap = [r["id"] for r in official_train if r["hash"] in test_hashes]
    splits = grouped_split([r for r in official_train if r["hash"] not in test_hashes],
                           {"train": .9, "validation": .1}, seed)
    splits["test"] = official_test
    return finish_manifest(folder, "cifar10", CIFAR_CLASSES, splits, seed, 32,
                           {"source": "https://www.cs.toronto.edu/~kriz/cifar.html",
                            "version": "CIFAR-10 python", "excluded_train_test_duplicates": overlap})


def export_cats(data_dir):
    """Use the official TFDS generator's corrupt-image filtering, then export PNGs."""
    folder = Path(data_dir) / "cats_vs_dogs"
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / "export_records.json"
    if record_path.exists() and (folder / "export_info.json").exists():
        return read_json(record_path)
    # TensorFlow must not reserve the RTX 2050: it is used only to prepare CPU data.
    import os
    import sys
    import tempfile
    import importlib
    import posixpath
    from types import SimpleNamespace
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if os.environ.get("ELL409_TF_LIBRARY"):
        sys.path.insert(0, os.environ["ELL409_TF_LIBRARY"])
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError("Cats preparation needs extras: pip install -e '.[cats]' (or use the cloud notebook).") from exc
    # TFDS download-cache names can exceed MAX_PATH beneath a long project path.
    cache = Path(os.environ.get("ELL409_TFDS_DIR", str(Path(tempfile.gettempdir()) / "ell409-tree-tfds"))) if os.name == "nt" else folder / "tfds"
    builder = tfds.builder("cats_vs_dogs:4.0.1", data_dir=str(cache))
    # In TFDS 4.9.10 on Windows, normpath supplies backslashes to ZipFile.open
    # after ZipInfo converted them to forward slashes. Keep the OFFICIAL generator
    # and its filtering/recoding, changing only archive-path normalization locally.
    module = importlib.import_module(builder.__class__.__module__)
    original_os = getattr(module, "os", None)
    workaround = os.name == "nt" and original_os is not None
    if workaround:
        module.os = SimpleNamespace(path=SimpleNamespace(normpath=archive_name))
    try:
        builder.download_and_prepare()
    finally:
        if workaround:
            module.os = original_os
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    (folder / "images").mkdir(exist_ok=True)
    records = []
    for i, item in enumerate(tfds.as_numpy(dataset)):
        array = item["image"]
        path = f"images/{i:05d}.png"
        image = Image.fromarray(array).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        image.save(folder / path)
        records.append({"id": f"cats-{i:05d}", "path": path,
                        "label": int(item["label"]), "hash": image_hash(array), "input_hash": image_hash(image)})
        if (i + 1) % 1000 == 0:
            print(f"Exported {i + 1} Cats vs Dogs images", flush=True)
    write_json(record_path, records)
    write_json(folder / "export_info.json", {"tfds_version": tfds.__version__,
               "builder_version": str(builder.version), "examples": len(records),
               "windows_zip_path_workaround": workaround, "export_resolution": 128,
               "export_resize": "Pillow RGB bilinear; lossless PNG; equivalent to resizing in ImageSplit"})
    return records


def archive_name(name):
    import posixpath
    return posixpath.normpath(name.replace("\\", "/"))


def prepare_cats(data_dir, seed=409):
    folder = Path(data_dir) / "cats_vs_dogs"
    if (folder / "manifest.json").exists():
        if read_json(folder / "manifest.json")["seed"] != seed:
            raise ValueError("Existing split seed differs.")
        return folder / "manifest.json"
    records = export_cats(data_dir)
    records, excluded = exclude_conflicting_duplicates(records)
    write_json(folder / "duplicate-audit.json", {"excluded_groups": excluded,
               "policy": "Exclude all members of contradictory exact-input duplicate groups; do not relabel."})
    splits = grouped_split(records, {"train": .8, "validation": .1, "test": .1}, seed)
    return finish_manifest(folder, "cats_vs_dogs", ["cat", "dog"], splits, seed, 128,
                           {"source": "https://www.tensorflow.org/datasets/catalog/cats_vs_dogs",
                            "version": "TFDS cats_vs_dogs:4.0.1", **read_json(folder / "export_info.json"),
                            "excluded_conflicting_duplicate_groups": excluded,
                            "modeling_examples": len(records),
                            "conflict_policy": "Exclude whole contradictory duplicate groups without relabeling; preserve source export."})


def exclude_conflicting_duplicates(records):
    groups = defaultdict(list)
    for r in records:
        groups[r.get("input_hash", r["hash"])].append(r)
    excluded = [rs for rs in groups.values() if len({r["label"] for r in rs}) > 1]
    ids = {r["id"] for rs in excluded for r in rs}
    return [r for r in records if r["id"] not in ids], excluded


def open_record(folder, record, arrays=None):
    if "index" in record:
        if arrays is None:
            arrays = np.load(Path(folder) / "images.npy", mmap_mode="r")
        return Image.fromarray(arrays[record["index"]]).convert("RGB")
    with Image.open(Path(folder) / record["path"]) as image:
        return image.convert("RGB")


def finish_manifest(folder, name, classes, splits, seed, resolution, source):
    hashes = {s: {h for r in rs for h in (r["hash"], r.get("input_hash", r["hash"]))} for s, rs in splits.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if hashes[a] & hashes[b]:
            raise ValueError(f"Duplicate leakage between {a} and {b}")
    arrays = np.load(folder / "images.npy", mmap_mode="r") if (folder / "images.npy").exists() else None
    summed, squared, count = np.zeros(3), np.zeros(3), 0
    for r in splits["train"]:
        image = open_record(folder, r, arrays).resize((resolution, resolution), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3) / 255
        summed += pixels.sum(axis=0)
        squared += (pixels * pixels).sum(axis=0)
        count += len(pixels)
    mean = summed / count
    std = np.sqrt(np.maximum(squared / count - mean * mean, 1e-12))
    manifest = {"dataset": name, "classes": classes, "seed": seed, "resolution": resolution,
                "source": source, "splits": splits,
                "counts": {s: len(rs) for s, rs in splits.items()},
                "normalization": {"mean": mean.tolist(), "std": std.tolist(), "computed_on": "train"}}
    manifest["fingerprint"] = digest(manifest)
    write_json(folder / "manifest.json", manifest)
    print(f"Prepared {name}: {manifest['counts']}", flush=True)
    return folder / "manifest.json"


class ImageSplit(Dataset):
    def __init__(self, manifest_path, split, limit=None):
        self.path = Path(manifest_path)
        self.folder = self.path.parent
        self.manifest = read_json(self.path)
        self.records = self.manifest["splits"][split]
        if limit is not None:
            # For smoke runs only; the full workflow never sets this.
            self.records = self.records[:limit]
        self.arrays = None
        self.mean = torch.tensor(self.manifest["normalization"]["mean"])[:, None, None]
        self.std = torch.tensor(self.manifest["normalization"]["std"])[:, None, None]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        if "index" in r and self.arrays is None:
            self.arrays = np.load(self.folder / "images.npy", mmap_mode="r")
        size = self.manifest["resolution"]
        image = open_record(self.folder, r, self.arrays).resize((size, size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
        return (x - self.mean) / self.std, r["label"], index


def synthetic_manifest(data_dir, seed=409):
    """Small clearly labeled plumbing test, not a substitute for either assignment dataset."""
    folder = Path(data_dir) / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    images = []
    records = []
    for i in range(120):
        label = i % 2
        array = rng.integers(0, 40, (32, 32, 3), dtype=np.uint8)
        array[:, :, label] += 170
        images.append(array)
        records.append({"id": f"synthetic-{i}", "index": i, "label": label, "hash": image_hash(array)})
    np.save(folder / "images.npy", np.stack(images))
    return finish_manifest(folder, "synthetic", ["red", "green"],
                           grouped_split(records, {"train": .8, "validation": .1, "test": .1}, seed),
                           seed, 32, {"source": "generated smoke test; NOT assignment evidence", "version": "1"})
