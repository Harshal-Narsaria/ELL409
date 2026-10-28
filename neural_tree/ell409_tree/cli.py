from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import choose_device, environment, read_json


def main():
    parser = argparse.ArgumentParser(description="ELL409 neural error-correction tree experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Show the actual training environment")
    p = sub.add_parser("prepare")
    p.add_argument("dataset", choices=["cifar10", "cats_vs_dogs", "synthetic"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=409)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--stage", default="all", choices=["all", "cnn", "features", "trees", "sweep", "compress", "plots"])
    p.add_argument("--finalize", action="store_true", help="Lock all candidates then evaluate held-out test splits")
    p = sub.add_parser("visualize")
    p.add_argument("--manifest", required=True); p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("finalize")
    p.add_argument("--manifest", required=True); p.add_argument("--run-dir", required=True)
    p.add_argument("--config", required=True); p.add_argument("--device", default="auto")
    p = sub.add_parser("report")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--output", default="output/ELL409_Neural_Tree_Report.pdf")
    p.add_argument("--repository-url")
    p.add_argument("--final", action="store_true", help="Require full dataset studies, training target, and private repository URL")
    p = sub.add_parser("package")
    p.add_argument("--output", default="dist/ell409-neural-tree.zip")
    p = sub.add_parser("predict")
    p.add_argument("--deployment", required=True); p.add_argument("--image", required=True)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("epoch-history")
    p.add_argument("--run-dir", required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=300)
    p = sub.add_parser("review", help="Audit locked results and export reviewed tables without retraining")
    p.add_argument("--runs-root", default="runs"); p.add_argument("--output-dir", default="results")
    p = sub.add_parser("benchmark", help="Repeat inference timings on validation images after training")
    p.add_argument("--runs-root", default="runs"); p.add_argument("--output-dir", default="results")
    p.add_argument("--device", default="auto"); p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    p = sub.add_parser("compact", help="Export function-preserving compact versions of the smallest tested fitting models")
    p.add_argument("--runs-root", default="runs"); p.add_argument("--output-dir", default="results/models")
    p.add_argument("--device", default="auto")
    p = sub.add_parser("submission", help="Make a standalone code/results/report folder for GitHub upload")
    p.add_argument("--runs-root", default="runs"); p.add_argument("--output", default="github-upload/ELL409-neural-tree")
    p = sub.add_parser("figures", help="Regenerate scientific plots from saved measurements without training")
    p.add_argument("--runs-root", default="evidence"); p.add_argument("--output-dir", default="results")
    p = sub.add_parser("dense-study", help="Additional isolated train/validation capacity measurements")
    p.add_argument("--source-root", default="runs")
    p.add_argument("--study-dir", default="runs/dense-capacity-2026-10-08")
    p.add_argument("--output-dir", default="results/dense-capacity-2026-10-08")
    p.add_argument("--config", default="configs/dense-study.json")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(environment(), indent=2))
    elif args.command == "prepare":
        from .data import prepare_cifar, prepare_cats, synthetic_manifest
        function = {"cifar10": prepare_cifar, "cats_vs_dogs": prepare_cats, "synthetic": synthetic_manifest}[args.dataset]
        print(function(args.data_dir, args.seed))
    elif args.command == "run":
        from .experiments import pipeline
        pipeline(args.manifest, args.run_dir, read_json(args.config), choose_device(args.device), args.stage, args.finalize)
    elif args.command == "visualize":
        from .visualize import visualize_nodes
        visualize_nodes(args.manifest, args.run_dir, choose_device(args.device))
    elif args.command == "finalize":
        from .experiments import finalize_tests
        finalize_tests(args.manifest, args.run_dir, read_json(args.config), choose_device(args.device))
    elif args.command == "report":
        from .report import create_report
        create_report(args.runs_root, args.output, args.repository_url, args.final)
    elif args.command == "package":
        from .packaging import package_project
        print(package_project(args.output))
    elif args.command == "predict":
        import numpy as np
        import torch
        from PIL import Image
        from .models import ImageTreeClassifier, TreeClassifier
        from .common import load_checkpoint
        bundle = load_checkpoint(args.deployment)
        if "trunk_state" not in bundle:
            raise ValueError("This older deployment lacks CNN weights; regenerate it with the current feature cache")
        tree = TreeClassifier.from_spec(bundle["spec"])
        tree.load_state_dict(bundle["model"])
        device = choose_device(args.device)
        model = ImageTreeClassifier(tree, bundle).to(device).eval()
        with Image.open(args.image) as im:
            image = im.convert("RGB").resize((bundle["resolution"], bundle["resolution"]), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)[None].to(device) / 255
        with torch.no_grad():
            prediction = int(model.predict(x)[0])
        print(json.dumps({"label": bundle["class_names"][prediction], "class_index": prediction}))
    elif args.command == "epoch-history":
        from .epoch_history import root_epoch_history
        root_epoch_history(args.run_dir, choose_device(args.device), args.epochs)
    elif args.command == "review":
        from .review import write_review
        write_review(args.runs_root, args.output_dir)
    elif args.command == "benchmark":
        from .benchmarks import repeated_latency
        repeated_latency(args.runs_root, args.output_dir, choose_device(args.device), args.repetitions, args.iterations)
    elif args.command == "compact":
        from .submission import export_compact_models
        export_compact_models(args.runs_root, args.output_dir, choose_device(args.device))
    elif args.command == "submission":
        from .submission import build_upload_folder
        build_upload_folder(args.runs_root, args.output)
    elif args.command == "figures":
        from .experiments import plot_results
        from .review import write_review
        for dataset in ("cifar10", "cats_vs_dogs"):
            plot_results(Path(args.runs_root) / dataset)
        write_review(args.runs_root, args.output_dir)
    elif args.command == "dense-study":
        from .dense_study import dense_study
        dense_study(args.source_root, args.study_dir, args.output_dir, args.config,
                    choose_device(args.device), args.prepare_only)


if __name__ == "__main__":
    main()
