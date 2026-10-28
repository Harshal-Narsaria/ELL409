# V6: Assignment 2 (augmentation, voting, bias/variance, loss landscape)

This is the final study. The frozen CNN80 is the root, and only the correction nodes and the parent α, β, offset learn. See [docs/DESIGN.md](../docs/DESIGN.md) §6 and [docs/CODE_WALKTHROUGH.md](../docs/CODE_WALKTHROUGH.md).

| File | Contents |
| --- | --- |
| `train.py` | `Trainer.grow` (recursive growth), `fit_base` (one node), `fit_parent` (LP, then Adam fallback), `evaluate` |
| `data.py` | `augmented_pixels` (3 fixed views per image), CNN feature extraction per view, `training_rows` (`none`/`repeat`/`augmented`), `stratified_bootstrap` |
| `analysis.py` | `majority_vote`, `brier_decomposition` (bias/variance), `landscape` (2-D loss slices + gate flips), `summarize` |
| `run.py` | `make_plan` (166 configurations), `fit_job`, repeat selection from training data, one-time test evaluation |
| `deploy.py` | `ImageEnsemble`: image → shared CNN → 1 or 3 trees → vote. Also a command-line predictor. |
| `subset_capacity_study.py` | Noisy 4,000-image CIFAR width study (report §7) |
| `plots.py` | Figures from saved measurements |
| `common.py` | Imports node classes from V1 to V3, parameter counting |
| `config.json` | All study settings (widths, depths, epochs, sample fractions, seeds, landscape grid) |
| `pilot.json` | Small-subset pilot settings used to check the pipeline before the full run |
| `tests/` | 11 tests: augmentation, bootstrap, voting, Brier identity, resume, masks, deployment, landscape |

```bash
python run.py --hours 10       # resumable; rerun the same command to continue
python deploy.py ../models/assignment2/cifar10-augmented-vote-3.pt ../examples/cifar10_test_0_cat.png
```
