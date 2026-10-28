# V1: shared CNN + neural correction trees (Assignment 1 main model)

A full CNN (32/64/128 channels, global average pool → 128 features) is trained once and frozen. Its features are cached, and a correction tree per class is trained on them: width 64, depth ≤ 4, ≤ 31 nodes per tree.

| File | Contents |
| --- | --- |
| `ell409_tree/models.py` | `SharedCNN`, `CorrectionNode` (corrected score), `TreeClassifier` (one-vs-rest), `correction_targets` |
| `ell409_tree/train.py` | CNN training, feature caching, `fit_binary` (one node), `TreeTrainer.grow` (recursive growth + parent refit), margin-1 parent LP |
| `ell409_tree/experiments.py` | Pipeline: CNN → features → trees → width sweep → pruning → plots → one-time test evaluation |
| `ell409_tree/data.py` | CIFAR-10 / Cats vs Dogs download, duplicate-aware split manifests |
| `ell409_tree/visualize.py` | Grad-CAM and occlusion maps for individual nodes |
| `ell409_tree/dense_study.py`, `epoch_history.py`, `benchmarks.py` | Extra capacity, epoch and timing studies |
| `ell409_tree/report.py`, `review.py`, `submission.py`, `packaging.py` | Report/PDF and upload helpers only (no learning logic) |
| `configs/` | Run settings (`cifar10.json`, `cats_vs_dogs.json`, `dense-study.json`, `smoke.json`) |

Entry point: `python -m ell409_tree --help`, run from this folder.
