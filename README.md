# ELL409: neural correction trees and augmentation

This repository documents two linked assignments: the original neural-tree study and the follow-up on input augmentation, voting ensembles, bias/variance and loss landscapes. It includes unsuccessful attempts as well as positive results.

**Follow-up snapshot: 2026-10-20T11:21:48+05:30. 85/166 primary configurations have completed. The study is still running; this upload is not a final follow-up report.** Read the dates before comparing snapshots. A configuration is a saved combination of settings; continued epoch budgets are not independent training runs.

## Start here

1. [Experiment journey](docs/EXPERIMENT_JOURNEY.md): what we tried and why.
2. [Results and limitations](docs/RESULTS_AND_LIMITATIONS.md): measured outcomes and the double-descent evidence.
3. [Follow-up methods](docs/FOLLOWUP_METHODS.md): how the four new questions are implemented.
4. [Figure guide](docs/FIGURE_GUIDE.md): which figures answer each question.
5. [Reproduction](docs/REPRODUCE.md): setup, tests, image inference and saved-data plots.

The [previous assignment's PDF](report/ELL409_Previous_Assignment_Report.pdf) is preserved as historical work. It does not report the ongoing augmentation study. The current evidence is under [followup](followup/RESULTS.md), including [all completed measurements](followup/measurements.csv).

## Initial follow-up comparison

| Dataset | Training images | Canonical train % | Validation % | Test % |
| --- | --- | ---: | ---: | ---: |
| cats_vs_dogs | augmented | 83.74 | 81.30 | 81.64 |
| cats_vs_dogs | none | 84.25 | 81.47 | 81.17 |
| cats_vs_dogs | repeat | 86.50 | 80.91 | 80.87 |
| cifar10 | augmented | 90.67 | 77.42 | 76.69 |
| cifar10 | none | 94.46 | 76.50 | 76.13 |
| cifar10 | repeat | 99.26 | 74.78 | 74.44 |

Here `repeat` means three copies of each original image, and `augmented` means the original plus two crop/flip views. Canonical training accuracy uses original images for a common comparison; interpolation must instead use errors on all actual training views. These are fixed width-64, depth-one reference models, not the deepest or largest models.

![CIFAR reference comparison](followup/figures/cifar10-reference-comparison.png)

Augmentation modestly improves initial test accuracy on both datasets. Deeper models demonstrate memorization, but the completed depth curves do not establish double descent. Earlier 15%-label-noise results show a suggestive full curve and repeatable improvement beyond interpolation; that later improvement largely came from wider root classifiers. [Read the qualifications](docs/RESULTS_AND_LIMITATIONS.md).

## Browse the implementation

| Folder | Purpose |
| --- | --- |
| `neural_tree/ell409_tree/` | Original shared CNN, recursive correction nodes, fitting, interpretation and pruning. |
| `neural_tree_v2/v2_tree/` | Pixel and selective local-CNN trees, fixed-root controls. |
| `neural_tree_v3/` | Partly trained CNN features, fixed CNN roots and parent-fitting refinement. |
| `neural_tree_v4/` | Frozen random-CNN features. |
| `neural_tree_v5/` | Clean/noisy-label width studies and longer-budget continuation. |
| `neural_tree_v6/` | Current input-augmentation, majority-vote, bias/variance and landscape study. |
| `followup/` | Timestamped completed measurements and graphs. |
| `figures/` | Selected historical noise and feature-design figures. |
| `archives/` | Original complete submission and supplementary follow-up evidence/model weights. |

The core source is visible without opening ZIPs. Extra historical configurations, tests, splits and evidence remain archived to keep fewer than 100 upload files. Raw images, feature banks and transient optimizer checkpoints are omitted. This snapshot is not a replacement for the local training workspace and is not a resume checkpoint for its unfinished job.

Repository: [Harshal-Narsaria/ELL409](https://github.com/Harshal-Narsaria/ELL409). This folder is prepared locally; upload, repository visibility, instructor access and course submission are separate steps.

Run `python tools/verify_repository.py` to check this snapshot. The [package index](PACKAGE_INDEX.json) records hashes, archived contents, source provenance and the captured training status.
