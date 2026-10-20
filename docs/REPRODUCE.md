# Setup, verification and reproduction

Commands are run from the repository root. The source files visible here are a reviewable subset of the historical project plus the complete follow-up implementation. The original archives retain the other tests, configs, manifests and results.

## Read and verify without datasets

```powershell
python tools/verify_repository.py
```

Use Python 3.12. Install a compatible PyTorch/torchvision pair for the target CPU or GPU, then:

```powershell
python -m pip install -r requirements.txt -r neural_tree_v6/requirements.txt
python -m pip install -e ./neural_tree -e ./neural_tree_v2
python -B -m pytest neural_tree_v6/tests -q -p no:cacheprovider
```

The exact recorded follow-up versions are in `neural_tree_v6/requirements-lock.txt`; a CUDA-specific torch build may require the platform's package index. These small tests require no raw dataset. Do not launch full training merely to inspect this submission.

## Restore follow-up evidence and regenerate graphs

```powershell
python tools/verify_repository.py --restore-followup
python -B neural_tree_v6/plots.py --run-dir neural_tree_v6/runs/augmentation-study --output followup
```

The restore command checks archive hashes and refuses to overwrite changed files. It restores only completed metrics, validation predictions, experiment definitions, landscape arrays and available deployment weights. It does not restore the unfinished worker's optimizer state or raw images. Regeneration recomputes voting and decomposition from saved predictions. Rebuilding outputs can change file bytes; keep the original upload snapshot for its hash verification.

## Predict a new image with an included model

After restoration:

```powershell
python neural_tree_v6/deploy.py followup/models/cats_vs_dogs-augmented-single.pt PATH_TO_IMAGE.jpg
```

The saved deployment includes its frozen CNN, correction model and normalization. Available three-member models use `*-vote-3.pt`. Models are trusted local coursework artifacts; do not load unrelated untrusted checkpoints.

## Historical reproduction and data preparation

Download both `archives/project-01.zip` and `archives/project-02.zip`, and extract both into the same separate empty folder, for example `historical-restored`. That restores the previous complete project with its own `REPRODUCE.md`, configs, split manifests, tests, report tools and selected deployments. Its original index is `archives/PREVIOUS_SUBMISSION_INDEX.json`. Separating this restoration avoids overwriting the source snapshot used for the follow-up.

Prepare the original datasets using those instructions and compare regenerated IDs and normalization against the saved manifests. Run the V3 CNN80 preparation before V6: the follow-up's `feature_source` expects `features.pt`, `pixels.pt` and `cnn-frozen.pt` under `neural_tree_v3/runs/cnn80-depth/<dataset>/`. Its optional noise controls reuse the V5 corruption manifests. Preserve versions, seeds, splits and transformations when moving hardware; feature hashes can differ when regenerating artifacts.

For a new full study, use a separate prepared workspace and a new run/output directory:

```powershell
python neural_tree_v6/run.py --config neural_tree_v6/config.json --run-dir neural_tree_v6/runs/new-study --output neural_tree_v6/results/new-study --hours 10
```

The hours flag is a configurable compute allowance. The local original worker has its own separately agreed cutoff; changing these copied files does not control it. Matching checkpoint recovery restores RNG and optimizer states; changed definitions are rejected. The upload snapshot is not enough to resume the unfinished local worker.

## Update this upload after training advances

From the original workspace:

```powershell
python tools/refresh_repository.py --workspace PATH_TO_ORIGINAL_WORKSPACE --destination PATH_TO_A_NEW_UPLOAD_FOLDER
```

The builder reads completed files only and requires a fresh destination. It keeps the old snapshot and training sources unchanged, regenerates plots from copied evidence, checks the package, and records a new capture time. A running job is never relabelled as complete.

## Short references

The supplied teaching manuscript defines the recursive correction masks. Related methods: [Nakkiran et al., Deep Double Descent](https://arxiv.org/abs/1912.02292), [Breiman, Bagging Predictors](https://www.stat.berkeley.edu/~breiman/bagging.pdf), and [Li et al., Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913). The historical report contains its bibliography and glossary.
