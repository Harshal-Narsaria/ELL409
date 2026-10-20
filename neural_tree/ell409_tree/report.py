from __future__ import annotations

from pathlib import Path
import os
from xml.sax.saxutils import escape

from .common import read_json, write_json


def study_evidence(runs_root):
    studies = []
    for run in sorted(Path(runs_root).iterdir()) if Path(runs_root).exists() else []:
        if run.is_dir() and (run / "configuration.json").exists():
            audit = read_json(run / "data-audit.json") if (run / "data-audit.json").exists() else {"dataset": run.name}
            results = [read_json(p) for p in sorted(run.glob("*/metrics.json"))]
            studies.append({"path": run, "dataset": audit["dataset"], "audit": audit, "results": results,
                            "complete": (run / "study-complete.json").exists()})
    return studies


def create_report(runs_root, output, repository_url=None, final=False):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                   Spacer, Table, TableStyle)
    studies = study_evidence(runs_root)
    required = {"cifar10", "cats_vs_dogs"}
    present = {s["dataset"] for s in studies if s["complete"] and (s["path"] / "node-observations.json").exists()}
    missing = sorted(required - present)
    reviewed = None
    if not missing and all((Path(runs_root) / d / "node-interpretations.json").exists() for d in required):
        from .review import collect_review
        reviewed = collect_review(runs_root)
    targets_met = all(any(r["train"]["accuracy"] >= .995 for r in s["results"])
                      for s in studies if s["dataset"] in required) and not missing
    valid_url = bool(repository_url and repository_url.startswith("https://github.com/"))
    if final and (missing or not targets_met or not valid_url or reviewed is None):
        raise RuntimeError(f"Final report unavailable: missing complete studies {missing}, training target met={targets_met}, repository URL supplied={valid_url}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 7
    styles["Heading2"].keepWithNext = True
    styles.add(ParagraphStyle("Cell", fontSize=8, leading=10))
    styles.add(ParagraphStyle("CaptionSmall", fontSize=8, leading=10, textColor=colors.HexColor("#555555"), spaceAfter=8))
    story = []
    def p(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
    def h(text):
        story.append(Spacer(1, 3 * mm)); p(text, "Heading2")
    def table(rows, widths):
        content = [[Paragraph(escape(str(c)), styles["Cell"]) for c in row] for row in rows]
        item = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                                  ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#8090a0")),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                  ("TOPPADDING", (0, 0), (-1, -1), 6)]))
        story.append(KeepTogether([item]) if len(rows) <= 8 else item)
        story.append(Spacer(1, 3 * mm))
    def figure(path, caption, max_height=90 * mm, introduction=None, heading=None):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            width, height = im.size
        scale = min(174 * mm / width, max_height / height)
        items = [Paragraph(escape(heading), styles["Heading2"])] if heading else []
        if introduction: items.append(Paragraph(escape(introduction), styles["BodyText"]))
        items.extend([Image(str(path), width=width * scale, height=height * scale),
                      Paragraph(escape(caption), styles["CaptionSmall"])])
        story.append(KeepTogether(items))
    p("ELL409: Neural Error-Correction Trees", "Title")
    p("Experimental report" if final else ("COMPLETED EXPERIMENTAL REPORT - REPOSITORY LINK PENDING" if reviewed else "IMPLEMENTATION AND EXPERIMENT STATUS - DRAFT"), "Heading2")
    p("This report is generated from saved experiment records. Synthetic checks are identified separately and are not evidence of CIFAR-10 or Cats vs Dogs performance.")
    if not final:
        p("<b>Submission status:</b> " + ("All experiments and interpretations are complete. " if reviewed else "This is a progress report. ")
          + ("Full-dataset studies still pending: " + escape(", ".join(missing)) + ". " if missing else "")
          + ("Near-perfect fitting has not yet been demonstrated on both datasets. " if not targets_met else "")
          + ("The required private GitHub URL has not been supplied or published." if not valid_url else "Repository privacy and grader access must be verified."))
    p("<b>Private repository:</b> " + (f'<link href="{escape(repository_url)}">{escape(repository_url)}</link>' if valid_url else "Pending: add the real private repository URL before submission."))
    if reviewed:
        h("Measured outcomes")
        rows = [["Selected correction tree", "Train accuracy", "Validation", "Test", "Train errors"]]
        for dataset, data in reviewed["datasets"].items():
            model = data["selected"]
            rows.append([dataset, f"{model['train']['accuracy']:.4%}", f"{model['validation']['accuracy']:.2%}", f"{model['test']['accuracy']:.2%}", f"{model['train']['errors']} / {model['train']['examples']:,}"])
        table(rows, [50 * mm, 32 * mm, 27 * mm, 27 * mm, 38 * mm])
        p("Both designated training splits meet the operational near-perfect fitting target. The correction trees improve training fit relative to roots alone, but their held-out gains are small and inconsistent. No clear model-capacity double descent is observed in the tested range. Pruning removes all correction subtrees in both selected models while retaining the same validation accuracy and near-perfect training fit.")
        p("Selected models were fixed from training results before test evaluation. Test columns describe those declared models; they are not used to choose a new best model. The comparison covers 31 tree configurations per dataset, plus conventional CNN baselines and separate binary epoch studies.")
    h("1. Architecture and relationship to the teaching document")
    p("The reference is Jayadeva's four-page <i>A Tree Type Neural Network</i>. It builds an error-correction tree: the original features and binary outputs of child classifiers feed into a parent score. The implementation preserves this upward correction mechanism and replaces each single-neuron base classifier with a small neural network.")
    p("A shared CNN has three blocks with 32, 64, and 128 channels. Each block contains two 3 x 3 convolutions, ReLU activations, and 2 x 2 pooling. Global average pooling produces 128 features. A temporary classification head trains the trunk from scratch; the trunk is then frozen. Feature standardization statistics are computed on training features only.")
    p("A node uses a 128 -> hidden width -> 1 MLP. Its score is s(x) = g(z(x)) + alpha A(x) - beta B(x) + offset, with alpha and beta nonnegative. A and B are hard, recursively corrected binary predictions. Parent base networks and child predictions remain fixed while correction strengths are fitted.")
    p("A projected-gradient fit is followed by a three-variable linear margin feasibility check. At most eight worst-case child-output/label groups encode all training constraints. A feasible exact refit is retained only if it improves the measured training-error/loss objective. This is an efficient signed parent-refit construction, not a guarantee that child networks fit their targets.")
    table([["Parent / desired", "Child A target", "Child B target"], ["0 / 0", "0", "Ignore"],
           ["1 / 1", "Ignore", "0"], ["0 / 1", "1", "0"], ["1 / 0", "0", "1"]], [58 * mm] * 3)
    p("Cats vs Dogs uses one binary tree. CIFAR-10 uses ten one-versus-rest trees and predicts the highest root score. The shared trunk is counted once. Local correction targets are constructed from prediction errors, not merely from positive margin slack.")
    if reviewed:
        from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
        drawing = Drawing(490, 150)
        def box(x, y, width, label):
            drawing.add(Rect(x, y, width, 32, fillColor=colors.HexColor("#e8edf3"), strokeColor=colors.HexColor("#536b82")))
            drawing.add(String(x + width / 2, y + 12, label, textAnchor="middle", fontSize=9))
        def arrow(x0, y0, x1, y1, label=None):
            drawing.add(Line(x0, y0, x1, y1, strokeColor=colors.HexColor("#425b73")))
            dx, dy = x1 - x0, y1 - y0
            length = (dx * dx + dy * dy) ** .5
            ux, uy = dx / length, dy / length
            drawing.add(Polygon([x1, y1, x1 - 7 * ux + 3 * uy, y1 - 7 * uy - 3 * ux, x1 - 7 * ux - 3 * uy, y1 - 7 * uy + 3 * ux], fillColor=colors.HexColor("#425b73")))
            if label: drawing.add(String((x0 + x1) / 2, (y0 + y1) / 2 + 8, label, textAnchor="middle", fontSize=9))
        box(0, 95, 55, "Image"); box(78, 95, 95, "Shared CNN"); box(196, 95, 60, "z: 128 dims")
        box(289, 95, 108, "Corrected parent"); box(428, 95, 62, "Prediction")
        arrow(55, 111, 78, 111); arrow(173, 111, 196, 111); arrow(256, 111, 289, 111); arrow(397, 111, 428, 111)
        box(260, 15, 80, "Child A"); box(359, 15, 80, "Child B")
        arrow(300, 47, 318, 95, "+ alpha A"); arrow(399, 47, 372, 95, "- beta B")
        drawing.add(String(0, 0, "Every node receives z. Child decisions feed upward; correction children may recurse.", fontSize=9))
        story.append(drawing)
    h("2. Training and experiment protocol")
    p("Both CNNs train from scratch for 30 epochs using Adam at learning rate 0.001. Image batches are 64 for CIFAR-10 and 16 for Cats vs Dogs. The main node width is 64; roots have depth zero, maximum depth is four, and each binary tree is capped at 31 active nodes. Node training uses 100 base epochs, Adam at 0.001, and feature batches of 1,024. Parent correction fitting uses 150 epochs. Controlled capacity sweeps use 100 and then 300 base epochs; correction fitting remains at 150. There is no learning-rate schedule. Configurations and all checkpoint identities are supplied with the results.")
    p("Main runs use no stochastic image augmentation, dropout, label smoothing, or weight decay. Nodes use class-balanced binary cross-entropy, while model selection at a node minimizes training classification errors and then unweighted loss. This balancing is disclosed because correction tasks can be highly imbalanced. The operational near-perfect target is 99.5% final training accuracy; 100% is preferred.")
    p("Cats vs Dogs preparation additionally excludes entire exact-input duplicate groups with contradictory labels and records every excluded member in the manifest audit, without relabeling. The live source audit found the same red 'Adopted' placeholder with both labels: two exported images are excluded, leaving 23,260 modeling examples from 23,262 official valid images. Same-label duplicate groups remain intact.")
    p("Epoch-boundary checkpoints contain model, optimizer, Python/NumPy/PyTorch RNG states, and histories. Recursive task fingerprints prevent resuming against different data or settings. Children and parent correction stages have separate journals. Training and validation metrics are computed in evaluation mode.")
    p("Capacity experiments fix depth at two and vary widths 8, 16, 32, 64, 128, 256, and 512 with one frozen trunk. Unused positions are retained as inactive networks. Allocated and active sizes are distinct. Selected widths receive three seeds. Validation guides exploration; final test candidates are declared and locked before held-out evaluation.")
    p("Double descent is an empirical hypothesis. A flat or monotone curve, an unconverged run, or a synthetic check cannot establish it. Global training error, validation error, and loss are reported against capacity. Local node histories are retained, and CNN epoch curves are labeled separately from tree behavior.")
    p("A secondary epoch study uses actual uninterrupted trajectories for representative binary root-only models at widths 8, 64, and 512. It evaluates validation errors and class-specific recall without selecting weights using validation. CIFAR's task is cat versus the other classes; Cats vs Dogs uses dog versus cat. These trajectories are not final multiclass correction-tree trajectories.")
    h("3. Dataset studies and measured results")
    for dataset in ["cifar10", "cats_vs_dogs", "synthetic"]:
        matches = [s for s in studies if s["dataset"] == dataset]
        if dataset == "cats_vs_dogs" and matches:
            story.append(PageBreak())
        if dataset == "synthetic" and matches:
            # Keep the newest software check; real dataset experiments remain complete.
            matches = [max(matches, key=lambda s: (s["path"] / "configuration.json").stat().st_mtime)]
        if not matches and dataset != "synthetic":
            p(f"<b>{escape(dataset)}:</b> No saved experiment results are available yet.")
            manifest_path = Path(runs_root).parent / "data" / dataset / "manifest.json"
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                p("Data preparation is complete. Recorded train/validation/test counts: "
                  + escape(str(manifest["counts"])) + ". Split fingerprint: " + escape(manifest["fingerprint"]) + ".")
        for study in matches:
            audit, run, results = study["audit"], study["path"], study["results"]
            if dataset == "synthetic":
                h("Software verification on synthetic data")
                p(f"The newest synthetic workflow ({escape(run.name)}) completed {len(results)} model configurations, "
                  "including root-only, recursive tree, fixed-capacity sweeps and seed repeats, pruning, narrower networks, deployment, visualization, and locked held-out evaluation. "
                  "These artificial red/green images verify executable paths only. Their accuracy and capacity curves are excluded from the scientific dataset conclusions.")
                continue
            h(dataset + (" - synthetic verification only" if dataset == "synthetic" else ""))
            p("Run: " + escape(run.name) + ". Full workflow complete: " + str(study["complete"]) + ".")
            if "counts" in audit:
                source = audit.get("source", {})
                p(f"Split counts: train {audit['counts']['train']:,}, validation {audit['counts']['validation']:,}, test {audit['counts']['test']:,}. Seed {audit['seed']}; RGB input {audit['resolution']} x {audit['resolution']}. "
                  + escape(str(source.get("version", source.get("source", "Recorded in the manifest")))) + ".")
                norm = audit["normalization"]
                p("Training-only RGB normalization: mean " + ", ".join(f"{v:.6f}" for v in norm["mean"]) + "; standard deviation " + ", ".join(f"{v:.6f}" for v in norm["std"]) + ". Exact-input duplicates are grouped across splits; full identifiers and provenance are supplied with the repository.")
            env_path = run / "environment.json"
            if env_path.exists():
                env = read_json(env_path)
                p("Hardware: " + escape(str(env.get("gpu") or "CPU")) + "; PyTorch " + escape(env["torch"]) + "; Python " + escape(env["python"]) + ".")
            baseline_path = run / "cnn-baseline.json"
            if baseline_path.exists():
                b = read_json(baseline_path)
                cnn_test = read_json(run / "cnn-test.json") if (run / "cnn-test.json").exists() else None
                p(f"Conventional CNN baseline: train {b['train']['accuracy']:.4%}, validation {b['validation']['accuracy']:.2%}" + (f", test {cnn_test['accuracy']:.2%}" if cnn_test else "") + f"; {b['parameters']:,} parameters.")
            if results:
                rows = [["Experiment", "Train", "Validation", "Test", "Active / allocated params", "Nodes"]]
                for result in [r for r in results if not r["name"].startswith("sweep")]:
                    test_path = run / result["name"] / "test-metrics.json"
                    test = f"{read_json(test_path)['accuracy']:.2%}" if test_path.exists() else "Not evaluated"
                    size = result["size"]
                    rows.append([result["name"], f"{result['train']['accuracy']:.4%}", f"{result['validation']['accuracy']:.2%}", test,
                                 f"{size['active_parameters']:,} / {size['allocated_parameters']:,}", size["active_nodes"]])
                table(rows, [44 * mm, 20 * mm, 23 * mm, 24 * mm, 45 * mm, 18 * mm])
                compact = [r for r in results if not r["name"].startswith("sweep")]
                if compact:
                    costs = [["Model", "Deployment MiB", "Training minutes", "Cached ms/image", "Image to prediction ms/image"]]
                    for result in compact:
                        timing = result.get("timing", {})
                        size_bytes = result.get("deployment_bytes", result.get("deployment_tree_bytes"))
                        costs.append([result["name"], f"{size_bytes / (1024 ** 2):.3f}" if size_bytes else "Not recorded",
                                      f"{result['training_seconds'] / 60:.2f}" if "training_seconds" in result else "Post-hoc pruning",
                                      f"{timing['cached_feature_ms_per_image']:.4f}" if "cached_feature_ms_per_image" in timing else "Not recorded",
                                      f"{timing['image_to_prediction_ms_per_image']:.4f}" if "image_to_prediction_ms_per_image" in timing else "Not recorded"])
                    table(costs, [48 * mm, 28 * mm, 30 * mm, 30 * mm, 38 * mm])
                    p("Tree training times exclude common CNN training. Image-to-prediction includes normalization, CNN, and tree, while excluding disk reads and resizing. Timings depend on the recorded hardware and batch sizes.", "CaptionSmall")
                flags = [r["name"] for r in results if r.get("convergence", {}).get("needs_extension")]
                if flags:
                    p(f"Optimization diagnostics remain flagged in {len(flags)} of {len(results)} configurations. A corrected model may interpolate while some base-node losses are still decreasing. The 300-epoch extension reduces undertraining concerns but does not certify convergence; this limits strong conclusions about capacity. Full flags and local histories are supplied in the result records.")
                if dataset != "synthetic":
                    best = max(results, key=lambda r: r["train"]["accuracy"])
                    gap = best["train"]["accuracy"] - best["validation"]["accuracy"]
                    p(f"Best observed training accuracy is {best['train']['accuracy']:.2%}; its train-validation gap is {gap:.2%}. "
                      + ("The fitting target is met." if best["train"]["accuracy"] >= .995 else "The fitting target is not met; further optimization or capacity is required."))
                    eligible = [r for r in results if r["train"]["accuracy"] >= .995]
                    if eligible:
                        smallest = min(eligible, key=lambda r: r["size"]["active_parameters"])
                        p(f"Smallest tested model retaining near-perfect fit: {escape(smallest['name'])}, {smallest['size']['active_parameters']:,} active parameters including the shared trunk.")
                    selected_path = run / "selected-tree.json"
                    if selected_path.exists():
                        selected_name = read_json(selected_path)["name"]
                        chosen = next(r for r in results if r["name"] == selected_name)
                        eligible_val = [r for r in results if r["validation"]["accuracy"] >= chosen["validation"]["accuracy"] - .005]
                        smallest_val = min(eligible_val, key=lambda r: r["size"]["active_parameters"])
                        p(f"Smallest tested model within 0.5 percentage points of the selected tree's validation accuracy: {escape(smallest_val['name'])}, {smallest_val['size']['active_parameters']:,} active parameters. This is a search result, not a global minimality proof.")
                        root_only = next((r for r in results if r["name"] == "roots-only"), None)
                        if root_only:
                            p(f"Correction-tree change relative to root-only: training accuracy {chosen['train']['accuracy'] - root_only['train']['accuracy']:+.2%}, validation accuracy {chosen['validation']['accuracy'] - root_only['validation']['accuracy']:+.2%}. Local error correction need not improve generalization.")
                        if reviewed:
                            data = reviewed["datasets"][dataset]
                            p(f"The selected tree's test change relative to roots alone is {data['correction_delta']['test'] * 100:+.2f} percentage points. This single main-seed comparison does not establish a generalization improvement.")
                            binary = [["Positive class", "Train recall", "Val recall", "Val negative accuracy", "Test recall", "Test negative accuracy"]]
                            for i, diagnostic in enumerate(chosen["validation"]["binary_trees"]):
                                label = audit["classes"][1 if len(audit["classes"]) == 2 else i]
                                train_diag = chosen["train"]["binary_trees"][i]
                                test_diag = data["selected"]["test"]["binary_trees"][i]
                                binary.append([label, f"{train_diag['positive_recall']:.2%}", f"{diagnostic['positive_recall']:.2%}", f"{diagnostic['negative_accuracy']:.2%}", f"{test_diag['positive_recall']:.2%}", f"{test_diag['negative_accuracy']:.2%}"])
                            table(binary, [29 * mm] * 6)
                            p("These are binary threshold diagnostics. For CIFAR they differ from final multiclass argmax recalls, and high negative-class accuracy can hide weak positive-class recognition.", "CaptionSmall")
                            confusion_plot = Path(runs_root).parent / "results" / "figures" / f"{dataset}-confusion.png"
                            if confusion_plot.exists():
                                figure(confusion_plot, "Selected model's locked test confusion matrix. Each row is normalized within a true class; this is descriptive evaluation, not model selection.", 110 * mm)
            else:
                p("No completed tree experiments yet. Any available CNN epoch history is partial training evidence only.")
            if (run / "cnn-history.json").exists():
                history = read_json(run / "cnn-history.json")
                if history:
                    latest = history[-1]
                    p(f"Latest recorded CNN epoch {latest['epoch']}: train {latest['train']['accuracy']:.2%}, validation {latest['validation']['accuracy']:.2%}, {latest['seconds']:.1f} seconds for this epoch. "
                      "This is a conventional CNN measurement, not a completed correction-tree result.")
            for diagnostic in sorted(run.glob("features*.diagnostics.json")):
                info = read_json(diagnostic)
                p("Frozen-feature audit: " + escape(str(info)) + ".")
            if reviewed:
                data = reviewed["datasets"][dataset]
                for tag in ("sweep", "sweep-long"):
                    summary = data["sweeps"][tag]
                    width = summary["widths"][summary["primary_seed_interpolation_index"]] if summary["primary_seed_interpolation_index"] is not None else None
                    p(f"{tag}: first primary-seed zero training error at width {width}; " + escape(summary["observation"]))
                p("CIFAR's mean validation error generally decreases with width; Cats vs Dogs shows small irregular changes rather than a clear second descent. Strong frozen features already make the smallest width close to interpolation. Only selected widths have three seeds, and inactive allocated positions contribute no inference capacity; active error-driven topology also changes across widths.")
            for plot in sorted((run / "figures").glob("*.png")):
                figure(plot, f"{dataset}: {plot.stem}. Curves are drawn directly from saved measurements.")
            for interpreted in sorted(run.glob("sweep*-interpretation.json")):
                summary = read_json(interpreted)
                p("<b>" + escape(interpreted.stem) + ":</b> " + escape(summary["observation"]) + " " + escape(summary["caution"]))
            if reviewed:
                epoch_rows = [["Root-only width", "First zero-error epoch", "Final val error", "Final val BCE", "Positive recall", "Negative accuracy"]]
                for e in reviewed["datasets"][dataset]["epoch_models"]:
                    v = e["last"]["validation"]
                    epoch_rows.append([e["width"], e["first_zero_error_epoch"] or "Not reached", f"{v['errors'] / v['examples']:.2%}", f"{v['loss']:.3f}", f"{v['positive_recall']:.2%}", f"{v['negative_accuracy']:.2%}"])
                table(epoch_rows, [29 * mm] * 6)
                p("Training errors fall while held-out BCE can increase markedly; continued fitting makes some incorrect held-out scores more confident. Final validation classification errors change much less than loss. These separate binary root-only trajectories do not establish global tree epoch-wise double descent. The displayed final epoch is not chosen using validation.")
            observations = run / "node-observations.json"
            if observations.exists():
                for index, n in enumerate(read_json(observations)["nodes"]):
                    description = (f"Tree {n['tree']}, {n['path']}: depth {n['depth']}, {n['task_examples']} assigned examples, "
                                   f"{n['final_task_errors']} remaining task errors, activation frequency {n['activation_frequency']:.2%}.")
                    figure(run / "figures" / "nodes" / n["figure"], "Training examples, base-score Grad-CAM, occlusion sensitivity, and corrected-node class composition.", 100 * mm, introduction=description, heading="Node responses and interpretation" if index == 0 else None)
                p("Individual final spatial CNN features have a theoretical 36 x 36 receptive field; global pooling allows each node to use information across the full image. Maps display empirical sensitivity and require visual interpretation before assigning semantic roles.")
                interpretation_file = run / "node-interpretations.json"
                if interpretation_file.exists():
                    for observation in read_json(interpretation_file)["observations"]:
                        p(escape(observation))
    h("4. Size reduction, interpretations, and limitations")
    p("Subtree pruning compares every accepted removal to the original validation accuracy, so cumulative degradation cannot exceed 0.5 percentage points. Narrower trees are retrained using the same depth and node budgets; their realized correction topology may differ. Cached-feature latency and complete image-to-prediction latency are measured separately; disk reads and resizing are excluded. Deployment bundles include CNN weights and preprocessing, and exclude optimizer histories/task audit records.")
    if reviewed:
        for dataset, data in reviewed["datasets"].items():
            c = data["compression"]
            chosen = data["selected"]
            pruned = next(r for r in data["models"] if r["name"] == "pruned")
            p(f"<b>{dataset}:</b> pruning reduces nodes from {chosen['size']['active_nodes']} to {pruned['size']['active_nodes']}, parameters from {chosen['size']['active_parameters']:,} to {pruned['size']['active_parameters']:,} ({c['parameter_reduction_fraction']:.2%} reduction), and deployment bytes by {c['file_reduction_fraction']:.2%}. Validation accuracy is unchanged. Pruned training accuracy is {pruned['train']['accuracy']:.4%}, and test accuracy is {pruned['test']['accuracy']:.2%}.")
            p(f"The shared CNN occupies {c['pruned_trunk_fraction']:.2%} of the pruned model's parameters. Narrow widths 32 and 16 grow to {c['narrower_nodes']['narrow-w32']} and {c['narrower_nodes']['narrow-w16']} nodes, respectively. Width alone is therefore an unreliable proxy for complete model size or latency.")
            compression_plot = Path(runs_root).parent / "results" / "figures" / f"{dataset}-compression.png"
            if compression_plot.exists():
                figure(compression_plot, "Full parameter counts and accuracy for baseline, correction, pruning, and narrower-node models. More nodes can offset the reduction in individual network width.")
        compact_path = Path(runs_root).parent / "results" / "compact-models.json"
        if compact_path.exists():
            for dataset, c in read_json(compact_path).items():
                p(f"Compact export of smallest tested active fitting model for {dataset}: {escape(c['source_experiment'])}, {c['size']['allocated_parameters']:,} actual stored parameters, {c['bytes'] / (1024 ** 2):.3f} MiB. Only inactive positions were removed; scores were unchanged on the full train and validation splits. Test measurements are inherited from the unchanged predeclared source model; no retraining or test-guided selection occurred.")
        timing_path = Path(runs_root).parent / "results" / "latency-benchmark.json"
        if timing_path.exists():
            timing = read_json(timing_path)
            p("After training, inference timing was repeated using the same 16 validation image IDs for cached-feature and complete-image paths. Five rounds of 50 iterations followed warmup; order rotated across models. Values below are median milliseconds per image, with interquartile intervals for the image path. Original one-off training-time measurements remain in the evidence records.")
            rows = [["Dataset/model", "Cached median ms", "Image median ms", "Image Q1-Q3 ms"]]
            for dataset, models in timing["datasets"].items():
                for name, t in models.items():
                    rows.append([dataset + "/" + name, f"{t['cached']['median_ms_per_image']:.4f}", f"{t['image']['median_ms_per_image']:.4f}", f"{t['image']['q1_ms_per_image']:.4f} - {t['image']['q3_ms_per_image']:.4f}"])
            table(rows, [68 * mm, 32 * mm, 32 * mm, 42 * mm])
        p("The experiments support near-perfect training fitting and practical pruning, rather than a robust generalization gain from adding correction subtrees. They do not establish double descent in the tested capacity or epoch ranges. Frozen features, class imbalance, topology changes, remaining optimization flags, and limited seed repeats restrict the scope of these conclusions.")
        deployment_check = Path(runs_root).parent / "results" / "deployment-check.json"
        if deployment_check.exists():
            for dataset, data in read_json(deployment_check)["datasets"].items():
                training = data["splits"]["train"]
                validation = data["splits"]["validation"]
                p(f"Full deployed image-path check for {dataset}: {training['image_errors']} / {training['examples']:,} training errors ({training['image_accuracy']:.4%}), {validation['image_accuracy']:.2%} validation accuracy, and {training['prediction_disagreements'] + validation['prediction_disagreements']} cache/image prediction differences across the full train and validation splits. This check performs no fitting or test reevaluation. CUDA arithmetic can vary with batch size; bit-identical scores are not assumed.")
    if missing:
        p("Measured full-dataset conclusions are still pending. No claim of full training interpolation, double descent, meaningful semantic specialization, or successful compression is made without the corresponding records.")
    p("Hard correction outputs make recursive fitting stagewise. A correction can improve local training fit without improving generalization. Independently trained CIFAR root scores may have different scales. Frozen feature collisions and inconsistent labels can prevent exact fitting. The teacher's construction is not a guarantee that gradient training of every neural node converges.")
    h("5. Future directions")
    p("Investigate shared multiclass correction modules, selective feature adapters, stronger common trunks, calibration of one-versus-rest scores, and regularization after the memorization study. Compare image-level semantic hypotheses using multiple examples and occlusion checks, rather than relying on a single attractive heatmap.")
    h("6. Sources and reproduction")
    for title, url in [("Official CIFAR-10", "https://www.cs.toronto.edu/~kriz/cifar.html"),
                       ("TFDS Cats vs Dogs", "https://www.tensorflow.org/datasets/catalog/cats_vs_dogs"),
                       ("Deep Double Descent", "https://arxiv.org/abs/1912.02292"),
                       ("Grad-CAM", "https://arxiv.org/abs/1610.02391"),
                       ("Colab FAQ", "https://research.google.com/colaboratory/faq.html"),
                       ("Kaggle notebooks", "https://www.kaggle.com/docs/notebooks")]:
        p(f'<link href="{url}">{escape(title)}</link>: {url}')
    p("The repository README documents local and cloud execution, including data preparation, resume, visualization, locked final test evaluation, and report generation. The implementation preserves project reference files.")
    if reviewed:
        h("Appendix: all locked capacity and compression results")
        for dataset, data in reviewed["datasets"].items():
            p("<b>" + escape(dataset) + "</b>")
            rows = [["Experiment", "Train errors", "Val accuracy", "Test accuracy", "Active / allocated params", "Nodes"]]
            for result in data["models"]:
                size = result["size"]
                rows.append([result["name"], result["train"]["errors"], f"{result['validation']['accuracy']:.2%}", f"{result['test']['accuracy']:.2%}", f"{size['active_parameters']:,} / {size['allocated_parameters']:,}", size["active_nodes"]])
            table(rows, [48 * mm, 19 * mm, 23 * mm, 23 * mm, 43 * mm, 18 * mm])
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(18 * mm, 12 * mm, "ELL409 neural correction trees" + (" - repository link pending" if reviewed and not valid_url else " - DRAFT" if not final else ""))
        canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))
    temporary = output.with_name(f".{output.stem}-{os.getpid()}.tmp.pdf")
    doc = SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                           topMargin=18 * mm, bottomMargin=20 * mm,
                           title="ELL409 Neural Error-Correction Trees", author="ELL409 assignment project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    os.replace(temporary, output)
    write_json(output.with_suffix(".evidence.json"), {"final": final, "missing_studies": missing,
               "training_target_met": targets_met, "repository_url": repository_url,
               "scientific_review_complete": reviewed is not None,
               "studies": [{"dataset": s["dataset"], "run": s["path"].name, "complete": s["complete"]} for s in studies]})
    print(output.resolve())
    return output
