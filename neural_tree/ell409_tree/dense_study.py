"""Isolated, resumable train/validation capacity measurements; no new test evaluation."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import copy
import csv
import hashlib
import json
import os
import shutil
import time

import numpy as np
import psutil

from .common import environment, load_checkpoint, read_json, tensor_hash, write_json
from .experiments import interpret_sweep, run_tree


def node_config(base, study, width, seed):
    return {**base, "width": width, "seed": seed, "node_epochs": study["node_epochs"],
            "correction_epochs": study["correction_epochs"], "max_depth": study["max_depth"],
            "max_nodes": study["max_nodes"], "fixed_topology": True}


def compatible(record, config, bundle):
    return (record.get("config") == config and record.get("features") == bundle["fingerprint"]
            and record.get("manifest") == bundle["manifest"])


def train_validation_bundle(bundle):
    value = dict(bundle)
    value["splits"] = {k: bundle["splits"][k] for k in ("train", "validation")}
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def fixed_definition(path, value):
    if Path(path).exists() and read_json(path) != value:
        raise ValueError("Study identity changed; choose a new study directory")
    write_json(path, value)


def jobs_for(dataset, study, source):
    jobs = {(w, study["primary_seed"]) for w in study["widths"]}
    jobs.update((w, s) for w in study["repeat_widths"][dataset] for s in study["repeat_seeds"])
    for path in Path(source).glob(study["source_prefix"] + "-w*-s*/metrics.json"):
        c = read_json(path)["config"]
        if c["width"] in study["widths"] and c["seed"] in study["repeat_seeds"]:
            jobs.add((c["width"], c["seed"]))
    return sorted(jobs, key=lambda ws: (ws[1] != study["primary_seed"], ws[0], ws[1]))


def summarize(records, study, expected):
    primary = sorted((r for r in records if r["config"]["seed"] == study["primary_seed"]), key=lambda r: r["config"]["width"])
    primary_complete = [r["config"]["width"] for r in primary] == study["widths"]
    complete = primary_complete and len(records) == expected
    s = interpret_sweep(primary, records) if primary else {"candidate_double_descent": False}
    s.update({"primary_complete": primary_complete, "all_repeats_complete": complete,
              "completed_primary_widths": [r["config"]["width"] for r in primary], "uses_test_results": False,
              "status": "complete" if complete else "partial; configurations or repeats pending",
              "models_with_optimization_flags": [r["name"] for r in records if r["convergence"]["needs_extension"]],
              "limitations": "Qualitative criterion, not a significance test. Selected widths have repeated seeds; active topology varies; remaining optimization flags limit capacity claims. This is exploratory validation evidence after prior test results were viewed."})
    if not complete:
        s["candidate_double_descent"] = False
        s["observation"] = "Additional configurations or repetitions are pending; no final interpretation yet."
    return s


def plot_curve(records, summary, study, dataset, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = sorted((r for r in records if r["config"]["seed"] == study["primary_seed"]), key=lambda r: r["config"]["width"])
    if not primary: return
    first = next((r for r in primary if r["train"]["errors"] == 0), None)
    for axis_type in ("width", "parameters"):
        def xof(r): return r["config"]["width"] if axis_type == "width" else r["size"]["allocated_parameters"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for i, split in enumerate(("train", "validation")):
            for j, metric in enumerate(("error", "loss")):
                def yof(r): return r[split]["errors"] / r[split]["examples"] * 100 if metric == "error" else r[split]["loss"]
                ax = axes[i, j]
                ax.plot([xof(r) for r in primary], [yof(r) for r in primary], "o-", label="Primary seed 409", color="#2166ac")
                groups = [[r for r in records if r["config"]["width"] == p["config"]["width"]] for p in primary]
                groups = [rs for rs in groups if len(rs) > 1]
                if groups:
                    ax.errorbar([xof(rs[0]) for rs in groups], [np.mean([yof(r) for r in rs]) for rs in groups],
                        yerr=[np.std([yof(r) for r in rs], ddof=1) for rs in groups], fmt="D", color="#d95f02", capsize=3,
                        label="Repeated-seed mean ± SD")
                if first: ax.axvline(xof(first), color="gray", linestyle="--", label=f"First primary zero error: w{first['config']['width']}")
                ax.set_xscale("log", base=2 if axis_type == "width" else 10)
                if axis_type == "width":
                    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                    ax.set_xticks(ticks, [str(t) for t in ticks]); ax.set_xticks([], minor=True)
                else:
                    positions = np.geomspace(min(xof(r) for r in primary), max(xof(r) for r in primary), 4)
                    ax.set_xticks(positions, [f"{p / 1e6:.3g}M" for p in positions])
                    ax.set_xticks([], minor=True)
                ax.set_xlabel("Node hidden width" if axis_type == "width" else "Allocated parameters, shared CNN counted once")
                ax.set_ylabel("Classification error (%)" if metric == "error" else "Cross-entropy")
                ax.set_title(split.title()); ax.grid(alpha=.2); ax.legend(fontsize=7); ax.margins(y=.15)
        label = "complete" if summary["all_repeats_complete"] else "partial; additional widths/repeats pending"
        fig.suptitle(f"{dataset}: {study['node_epochs']} base epochs; {label}")
        fig.tight_layout(rect=(0, .045, 1, .96))
        fig.text(.5, .014, "Measured runs only. Selected widths have repeated seeds; SD bars are not confidence intervals. No new test evaluation.", ha="center", fontsize=9)
        fig.savefig(output / f"{dataset}-capacity-{axis_type}.png", dpi=180); plt.close(fig)


def export_results(study_dir, output, study):
    output.mkdir(parents=True, exist_ok=True)
    rows, summaries, all_records = [], {}, {}
    definition = read_json(study_dir / "definition.json")
    for dataset in study["datasets"]:
        records = [read_json(p) for p in (study_dir / dataset).glob("dense-w*-s*/metrics.json")]
        records.sort(key=lambda r: (r["config"]["width"], r["config"]["seed"]))
        all_records[dataset] = records
        summaries[dataset] = summarize(records, study, len(definition["datasets"][dataset]["jobs"]))
        for r in records:
            c, tr, va, size = r["config"], r["train"], r["validation"], r["size"]
            rows.append({"dataset": dataset, "width": c["width"], "seed": c["seed"], "node_epochs": c["node_epochs"],
                "train_errors": tr["errors"], "train_examples": tr["examples"], "train_error_percent": tr["errors"] / tr["examples"] * 100,
                "validation_errors": va["errors"], "validation_examples": va["examples"], "validation_error_percent": va["errors"] / va["examples"] * 100,
                "train_loss": tr["loss"], "validation_loss": va["loss"], "active_parameters": size["active_parameters"],
                "allocated_parameters": size["allocated_parameters"], "active_nodes": size["active_nodes"],
                "reused": r.get("dense_provenance", {}).get("reused", False), "optimization_flag": r["convergence"]["needs_extension"]})
        if records: plot_curve(records, summaries[dataset], study, dataset, output)
    if rows:
        temp = output / "measurements.csv.tmp"
        with temp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        os.replace(temp, output / "measurements.csv")
    write_json(output / "interpretation.json", summaries)
    write_json(output / "measurements.json", all_records)
    write_json(output / "study.json", study)
    text = ["# Dense validation-only capacity study", "", study["policy"], "",
        f"Widths: {study['widths']}. Base networks: {study['node_epochs']} epochs; corrections: {study['correction_epochs']}. Frozen original CNN; depth-two allocation, seven positions per binary tree.", "",
        "Matching original long-sweep records are reused with exact configuration/feature identity and provenance. New widths start from scratch with epoch-boundary resume. Measured points are connected; diamonds and bars show repeated-seed means and sample standard deviations. First primary zero error is marked. No spline interpolation or new test evaluation.", ""]
    for d, s in summaries.items():
        text += [f"## {d}", "", s["status"], "", s.get("observation", "No points yet."), "",
            f"Completed primary widths: {s['completed_primary_widths']}. Primary zero-error index: {s.get('primary_seed_interpolation_index')}.", "",
            f"Models with remaining optimization flags: {len(s['models_with_optimization_flags'])}.", "", s["limitations"], ""]
    text += ["The qualitative change threshold is max(0.5 percentage points, twice the relevant standard error). It is a screening rule, not a significance test. If width one already interpolates, a separately declared weaker-feature study would be needed to sample underfitting. Width and parameter-count versions of each plot are supplied; allocated/active size is in the table. Original selected models and PDF remain unchanged.", ""]
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")
    return all_records


def protected_snapshot(source_root, project):
    files = []
    for dataset in ("cifar10", "cats_vs_dogs"):
        run = source_root / dataset
        files += list(run.glob("sweep*.json")) + list(run.glob("*/test-metrics.json"))
        files += [p for p in (run / "test-declaration.json", run / "selected-tree.json", run / "study-complete.json", run / "configuration.json", run / "cnn-test.json") if p.exists()]
    files += [p for p in (project / "output/ELL409_Neural_Tree_Report.pdf", project / "results/model-comparison.csv") if p.exists()]
    return {str(p.resolve()): file_hash(p) for p in files}


def package_extension(study_dir, output, project):
    archive = output / "full-records.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for path in sorted(study_dir.rglob("*.json")):
            if path.name not in ("runner-lock.json", "progress.json"): z.write(path, path.relative_to(study_dir).as_posix())
    with ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError("Dense records archive integrity failure")
    source, target = project / "github-upload/ELL409-professor-view", project / "github-upload/ELL409-professor-view-dense"
    if not source.exists(): return None
    if target.exists() and any(target.iterdir()):
        previous = read_json(target / "UPLOAD_MANIFEST.json")
        for row in previous["files"]:
            if file_hash(target / row["path"]) != row["sha256"]:
                raise ValueError("Existing dense upload copy was edited; preserve it")
        if file_hash(target / "results/dense-study/measurements.csv") != file_hash(output / "measurements.csv"):
            raise ValueError("Existing dense upload belongs to different measurements")
        return str(target)
    shutil.copytree(source, target, dirs_exist_ok=True)
    for path in (project / "ell409_tree").glob("*.py"): shutil.copy2(path, target / "ell409_tree" / path.name)
    for relative in ("configs/dense-study.json", "scripts/run-dense-study.ps1", "docs/DENSE_STUDY.md", "tests/test_dense_study.py"):
        shutil.copy2(project / relative, target / relative)
    shutil.copytree(output, target / "results/dense-study")
    with (target / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n## Additional dense validation study\n\nSee [dense results](results/dense-study/README.md) and [reproduction guide](docs/DENSE_STUDY.md). The original PDF/test results remain unchanged; these curves are an exploratory validation addendum.\n")
    manifest = [{"path": p.relative_to(target).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
                for p in sorted(target.rglob("*")) if p.is_file() and p.name != "UPLOAD_MANIFEST.json"]
    write_json(target / "UPLOAD_MANIFEST.json", {"files": manifest, "dense_addendum": "Validation only; original PDF/test results unchanged"})
    outer = project / "dist/ELL409-professor-view-dense.zip"
    with ZipFile(outer, "w", ZIP_DEFLATED) as z:
        for p in sorted(target.rglob("*")):
            if p.is_file(): z.write(p, target.name + "/" + p.relative_to(target).as_posix())
    with ZipFile(outer) as z:
        if z.testzip() is not None: raise ValueError("Dense upload archive integrity failure")
    return str(target)


def save_histories(folder):
    dest = folder / "histories.json"
    if dest.exists() or not (folder / "tree.pt").exists(): return
    saved = load_checkpoint(folder / "tree.pt")
    record = {c: {p: {"examples": len(t["labels"]), "positives": int(t["labels"].sum()),
                     "final_errors": t["final_errors"], "base": t["base"], "correction": t.get("correction")}
                  for p, t in ts.items()} for c, ts in saved["tasks"].items()}
    write_json(dest, record)


def dense_study(source_root, study_dir, output, config_path, device, prepare_only=False):
    source_root, study_dir, output = Path(source_root).resolve(), Path(study_dir).resolve(), Path(output).resolve()
    project = Path(__file__).resolve().parents[1]
    study = read_json(config_path)
    original_runs = [source_root / dataset for dataset in study["datasets"]]
    if study_dir == source_root or any(study_dir == p or study_dir in p.parents or p in study_dir.parents for p in original_runs):
        raise ValueError("New study must be separate from original dataset runs")
    if not study["widths"] or study["widths"] != sorted(set(study["widths"])) or study["widths"][0] < 1:
        raise ValueError("Widths must be ascending distinct positive integers")
    study_dir.mkdir(parents=True, exist_ok=True)
    lock = study_dir / "runner-lock.json"
    if lock.exists():
        old = read_json(lock)
        try:
            p = psutil.Process(old["pid"])
            if abs(p.create_time() - old["created"]) < .1 and p.is_running(): raise RuntimeError(f"Study already running: PID {p.pid}")
        except psutil.NoSuchProcess: pass
        lock.unlink()
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: raise RuntimeError("Another runner acquired the study lock")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, f)
    begin = time.time()
    status = {"status": "preparing", "pid": os.getpid(), "started_unix": begin, "completed": 0, "total": 0}
    try:
        contexts, definition = {}, {"study": study, "datasets": {}, "protected_originals": protected_snapshot(source_root, project)}
        for dataset in study["datasets"]:
            source = source_root / dataset
            bundle = train_validation_bundle(load_checkpoint(source / "features.pt"))
            base = read_json(source / "configuration.json")["config"]
            jobs = jobs_for(dataset, study, source)
            definition["datasets"][dataset] = {"manifest": bundle["manifest"], "features": bundle["fingerprint"],
                "trunk_hash": bundle["trunk_hash"], "validation_hash": tensor_hash(bundle["splits"]["validation"]["x"], bundle["splits"]["validation"]["y"]),
                "base_config": base, "jobs": jobs, "source": str(source)}
            contexts[dataset] = (source, bundle, base, jobs)
        definition = json.loads(json.dumps(definition))
        fixed_definition(study_dir / "definition.json", definition)
        write_json(study_dir / "environment.json", {"device": str(device), **environment()})
        status["total"] = sum(len(c[3]) for c in contexts.values())
        for dataset, (source, bundle, base, jobs) in contexts.items():
            for width, seed in jobs:
                name = f"dense-w{width}-s{seed}"
                dest = study_dir / dataset / name / "metrics.json"
                config = node_config(base, study, width, seed)
                if dest.exists():
                    record = read_json(dest)
                    if not compatible(record, config, bundle): raise ValueError("Existing dense result identity mismatch")
                    if "dense_provenance" not in record:
                        record["dense_provenance"] = {"reused": False, "policy": study["policy"]}; write_json(dest, record)
                    save_histories(dest.parent)
                    continue
                original = source / f"{study['source_prefix']}-w{width}-s{seed}/metrics.json"
                if original.exists() and compatible(read_json(original), config, bundle):
                    record = copy.deepcopy(read_json(original)); record["name"] = name
                    record["dense_provenance"] = {"reused": True, "source": str(original), "sha256": file_hash(original)}
                    write_json(dest, record)
        records = export_results(study_dir, output, study)
        status.update({"completed": sum(len(rs) for rs in records.values()), "status": "prepared" if prepare_only else "running"})
        status["new_runs_remaining"] = status["total"] - status["completed"]
        write_json(study_dir / "progress.json", status)
        print(f"Dense study: {status['total']} records, {status['completed']} available, {status['new_runs_remaining']} runs remaining", flush=True)
        if prepare_only: return status
        for dataset, (_, bundle, base, jobs) in contexts.items():
            for width, seed in jobs:
                name = f"dense-w{width}-s{seed}"
                dest = study_dir / dataset / name / "metrics.json"
                if dest.exists(): continue
                status.update({"dataset": dataset, "width": width, "seed": seed, "current": name, "updated_unix": time.time()})
                write_json(study_dir / "progress.json", status)
                print(f"Starting {dataset}/{name}: {status['completed']}/{status['total']} complete", flush=True)
                record = run_tree(bundle, study_dir / dataset, name, node_config(base, study, width, seed), device)
                record["dense_provenance"] = {"reused": False, "policy": study["policy"]}; write_json(dest, record)
                save_histories(dest.parent)
                status["completed"] += 1
                status["new_runs_remaining"] = status["total"] - status["completed"]
                status["elapsed_seconds"] = time.time() - begin
                write_json(study_dir / "progress.json", status)
                export_results(study_dir, output, study)
        for path, expected in definition["protected_originals"].items():
            if file_hash(path) != expected: raise ValueError("An original protected result changed: " + path)
        export_results(study_dir, output, study)
        upload = package_extension(study_dir, output, project)
        status.update({"status": "complete", "completed": status["total"], "elapsed_seconds": time.time() - begin,
                       "original_results_verified_unchanged": True, "upload_folder": upload, "uses_test_results": False})
        write_json(study_dir / "progress.json", status)
        print(f"Dense study complete. Results: {output}; upload copy: {upload}", flush=True)
        return status
    except Exception as exc:
        status.update({"status": "failed", "error": repr(exc), "updated_unix": time.time()})
        write_json(study_dir / "progress.json", status)
        raise
    finally:
        if lock.exists(): lock.unlink()
