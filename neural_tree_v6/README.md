# ELL409 follow-up: augmentation and neural correction trees

This is a separate version for the follow-up assignment shown on 19 October 2026. Earlier code, data, trained models and submission files are preserved. The screenshot lists a deadline of 23 October 2026, 11:59 PM; this has not been checked live on Moodle.

The supplied brief is preserved in [assignment.png](assignment.png).

## Models and experiments

| Assignment question | Implemented experiment |
| --- | --- |
| Augmentation, overfitting and double descent | Compare unaugmented and augmented correction trees across child widths and, separately, tree depths. Track exact training-view errors, validation error and later test error. Include original labels and a separate 15% label-noise control. |
| Multiple trees and majority voting | Train three bootstrap members; compare 1 and 3 complete classifiers voting on class labels. The code also supports a five-member profile. Each CIFAR member contains ten one-versus-rest trees. Those ten binary trees are one classifier, not ten ensemble votes. |
| Bias and variance versus samples and epochs | Stratified image-level bootstraps at 25%, 50% and 100% of the training-pool size; child-epoch budgets at 20, 100 and 400. Use three paired replicates at every sample size. |
| Loss landscape | Two normalized random weight directions around predeclared reference models, with final classification losses and hard-gate flip rates. |

Both **CIFAR-10 and Cats vs Dogs** use the earlier fixed splits. CIFAR has 45,000 training images, 5,000 validation images and 10,000 test images. Cats has 18,608 / 2,326 / 2,326. Both use the earlier 32 x 32 CNN80 checkpoint for this follow-up.

The CNN and its classification head are frozen. Root decisions therefore start with the earlier CNN near 80% training accuracy. Small correction MLPs learn residual mistakes using the original positive/negative child masks. Nonnegative correction strengths use the previously verified positive-margin LP where feasible and signed Adam otherwise. Root outputs feed upward through actual hard correction decisions. Depth comparisons keep the root and child width fixed; width comparisons keep the depth fixed.

## Augmentation that can still be fitted

Image augmentation happens **before** CNN extraction. For each training image, the main study caches three views: the original and two reflection-padded random crops with horizontal flips. A view's transform is keyed by the original image ID and view number, so interruption, batching and member order do not change it. Labels stay associated with the original image. Validation and test images are not augmented.

A fixed view bank provides stable correction targets and a meaningful finite training set whose interpolation can be measured. This is not an unlimited stream of new random augmentations. Three views are not three independent original images.

A repeated-canonical control at the reference width repeats each original image three times without transforming it. It matches input-view exposure and helps assess whether gains merely come from more training updates. Masked child-task sizes can differ, so actual node optimizer-update counts are logged; this is not claimed to be an exact step match.

Training uses no dropout, label smoothing or weight decay. Augmentation itself changes the learning problem and acts as a regularizer. It may improve generalization or postpone interpolation, and does **not** guarantee double descent. The existing 15% corruption map is reused for the separate noise control; every view of an original gets the same study label. Evaluation labels remain clean.

## What the bias–variance numbers mean

For a fixed validation image with observed one-hot label `y`, collect class-probability vectors `p_m` from independently fitted bootstrap members. With `p_bar = mean_m p_m`, report:

`mean_m ||p_m - y||^2 = ||p_bar - y||^2 + mean_m ||p_m - p_bar||^2`.

The terms are averaged over the same held-out images. Variance uses denominator `M`, so the empirical identity is exact. The first term is a **squared-bias proxy against observed labels**, not bias against unknown true class probabilities. It does not identify irreducible noise. A separate decomposition of hard one-hot member predictions is saved too.

These are **conditional** estimates: the CNN was previously trained on the full clean training pool and is shared. Sample-size changes refer to images used to train correction trees; this is not an experiment retraining the entire CNN on only that many images. Bootstrap draws contain duplicates, so draw counts and unique-original counts are both recorded. All views of a sampled original travel together, with bootstrap multiplicity retained.

An epoch is one pass through a child's assigned task. The depth-one epoch study has fixed root decisions and fixed child targets; identical tasks resume their optimizer and RNG states at later budgets. A changed deeper correction task starts fresh. These are common **per-child** epoch budgets, not epochs of end-to-end CNN training.

Nodes optimize class-balanced binary cross-entropy. At each budget, their checkpoint is selected by the fewest assigned training errors, then lower unweighted loss. A selected checkpoint can precede the last allowed epoch, so the epoch-axis is a training-budget comparison. This criterion can favor conservative corrections on rare-positive tasks; parent refitting and the final full-classifier metrics determine their actual effect.

An initial full-data width-32/200-epoch Cats control stayed near its frozen root: 83.72% training accuracy. Reusing the exact same learned bases at their final epoch changed only five predictions (83.74%), so checkpoint selection alone did not solve the bottleneck. Those controls are preserved under `runs/initial-200epoch-control` and `results/initial-200epoch-control`. The overnight profile therefore uses 400 epochs, widths through 512 and depth limits through 12 with at most 127 nodes per binary tree. Early low-budget sample/variance checks and stronger/deeper models are prioritized before exhaustive repeats. This revision was based on training diagnostics, not test scores.

## Interpreting landscapes and generalization

Loss slices perturb output-row-normalized MLP weight matrices while holding the CNN, biases and correction strengths fixed. Actual hard gates are recomputed, and a gate-flip map helps explain jumps. These are two local parameter-space directions, not the full landscape. Final classification loss is plotted; training uses separate masked node objectives, so the saved model is not guaranteed to be stationary for that plotted global loss. No smooth Hessian, global-convexity or global-minimum claim is made.

Model choices and repeats use training diagnostics, never test scores. Validation is available during exploration. The old test partitions have already been used in the earlier assignment, so the later test results are exploratory. The fixed six-model reference comparison is evaluated once all six predeclared models finish; the rest wait for the full primary plan and training-selected repeats. Later model settings are already fixed before reference test results are seen. Debug runs never evaluate test data. Time-limited runs retain completed evidence and checkpoints without pretending the entire study is complete.

## Run and resume

From the project root, use the existing environment:

```powershell
.venv/Scripts/python.exe -B -m pytest neural_tree_v6/tests -q -p no:cacheprovider
.venv/Scripts/python.exe -B -u neural_tree_v6/run.py --hours 10
```

The same command resumes completed jobs and matching checkpoints. The OS-held lock prevents duplicate workers in the same run directory. Resume refuses changed source/configuration definitions; use a new run directory for a changed experiment.

For the small integration pilot:

```powershell
.venv/Scripts/python.exe -B -u neural_tree_v6/run.py --config neural_tree_v6/pilot.json --run-dir neural_tree_v6/runs/pilot --output neural_tree_v6/results/pilot --hours 0.5
```

Pilot results use balanced subsets and tiny budgets; they verify the pipeline, not full-dataset assignment performance.

Progress: `runs/augmentation-study/progress.json`. Completed measurements, saved-data plots, bias–variance tables, ensemble comparisons, deployable models and `RESULTS.md` live under `results/augmentation-study/`. Reference-model test columns appear after their fixed comparison finishes; remaining tests wait for final evaluation. A 10-hour limit is a compute cap, not a promise that every configuration finishes.

Width controls use 16 through 512 hidden units; the noise control uses the predeclared overlapping widths 32/128/512. Depth limits do not automatically imply larger effective models: both allocated and active parameter counts are saved. Repeated epoch checkpoints are configurations, not independent training replicates.

```powershell
.venv/Scripts/python.exe -B neural_tree_v6/plots.py
.venv/Scripts/python.exe -B neural_tree_v6/deploy.py MODEL.pt IMAGE.jpg
```

On Windows the worker temporarily requests that the system remain awake while it runs; the request is released on exit. Keep the laptop plugged in with the lid open. Closing the lid, hibernation, shutdown or stopping the process can still interrupt training. Checkpoints support resumption; training does not run while the computer is asleep.

## Files

- `config.json`, `pilot.json`: declared experiments and separate integration-test settings.
- `data.py`: reproducible image-view features and grouped bootstrap sampling.
- `train.py`: recursive fitting, checkpoint recovery and exact-task epoch continuation.
- `analysis.py`: voting, empirical decomposition and loss/gate landscapes.
- `deploy.py`: image prediction with one shared CNN and one or more classifiers.
- `run.py`: finite training queue, status, recovery and final test evaluation.
- `plots.py`: measured-data figures, CSV tables and a readable results summary.
- `common.py`: shared model imports, parameter accounting and configuration checks.
- `tests/`: checks of the experiments' mathematical and implementation assumptions.

The main adaptations are the fixed augmentation bank for hard correction tasks, the repeated-image exposure control, grouped paired bootstrap comparisons, and loss slices accompanied by gate-flip maps. These adapt established methods; no claim of a new general learning algorithm is made.

## Short references

1. Supplied *Tree Type Neural Network* teaching manuscript: error-correction masks and recursive structure.
2. [Nakkiran et al., Deep Double Descent](https://arxiv.org/abs/1912.02292): capacity, sample-size and training-time phenomena.
3. [Breiman, Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf): resampled predictors and aggregation.
4. [Li et al., Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913): normalized parameter directions; adapted here to MLP output rows.
5. [Torchvision transformation documentation](https://docs.pytorch.org/vision/stable/transforms.html): standard crop and horizontal-flip image transformations.
