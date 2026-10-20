"""Build a reviewable repository snapshot without touching a running experiment.

Run in the original workspace, or supply --workspace. A new destination is
required on each invocation: an earlier upload is never silently replaced.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED


IST = timezone(timedelta(hours=5, minutes=30))
LIMIT = 24 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + '\n', encoding='utf-8')


def copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def archive_parts(files, destination, prefix):
    """Each source is independently checked; never split a ZIP into unusable chunks."""
    entries = []
    parts = []
    current = None
    for source, relative in files:
        if current is None:
            name = f'{prefix}-{len(parts)+1:02d}.zip'
            current = ZipFile(destination / name, 'w', ZIP_DEFLATED, compresslevel=5)
            parts.append(name)
        # Raw-size budgeting deliberately errs on the small side.
        if current.namelist() and current.fp.tell() + source.stat().st_size > LIMIT:
            current.close()
            current = None
            name = f'{prefix}-{len(parts)+1:02d}.zip'
            current = ZipFile(destination / name, 'w', ZIP_DEFLATED, compresslevel=5)
            parts.append(name)
        digest = sha(source)
        current.write(source, relative)
        entries.append({'archive': parts[-1], 'path': relative,
                        'bytes': source.stat().st_size, 'sha256': digest})
        assert sha(source) == digest, f'Source changed during archive copy: {source}'
    if current:
        current.close()
    return entries


VERIFY = r'''"""Verify a snapshot; optionally restore only its follow-up plotting inputs."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(data): return hashlib.sha256(data).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-followup', action='store_true')
    a = parser.parse_args()
    manifest = json.loads((ROOT/'PACKAGE_INDEX.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
              and '.git' not in p.parts}
    expected = {x['path'] for x in manifest['files']} | {'PACKAGE_INDEX.json'}
    for row in manifest['files']:
        path = ROOT/row['path']
        assert path.is_file(), f'Missing: {row["path"]}'
        assert path.stat().st_size == row['bytes'], f'Size changed: {path}'
        assert digest(path.read_bytes()) == row['sha256'], f'Hash changed: {path}'
    extra = actual - expected
    if extra: print(f'{len(extra)} additional local files (for example restored evidence); not part of the upload snapshot.')
    archived = {}
    for row in manifest['followup_archive_entries']:
        archived.setdefault(row['archive'], []).append(row)
    for name, rows in archived.items():
        with ZipFile(ROOT/'archives'/name) as archive:
            assert archive.testzip() is None, f'Corrupt ZIP: {name}'
            assert set(archive.namelist()) == {r['path'] for r in rows}
            for row in rows:
                value = archive.read(row['path'])
                assert len(value) == row['bytes'] and digest(value) == row['sha256']
                if a.restore_followup:
                    target = (ROOT/row['path']).resolve()
                    assert target.is_relative_to(ROOT.resolve()), 'Unsafe archive path'
                    if target.exists():
                        assert digest(target.read_bytes()) == row['sha256'], f'Refusing to replace changed local file: {target}'
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(value)
    for row in manifest['legacy_archives']:
        with ZipFile(ROOT/row['path']) as archive:
            assert archive.testzip() is None
            for member in archive.namelist():
                assert (ROOT/member).resolve().is_relative_to(ROOT.resolve())
    assert len(expected) < 100
    assert all(row['bytes'] < 25*1024*1024 for row in manifest['files'])
    print(f'PASS: {len(expected)} upload files, hashes, archive contents and browser-upload sizes verified.')
    if a.restore_followup:
        print('Follow-up plotting inputs and saved image models restored. No training started.')

if __name__ == '__main__': main()
'''


def make_documents(dest, snapshot, records):
    completed = len(records)
    label = snapshot['captured_ist']
    refs = [r for r in records if r['job']['sampler'] == 'full'
            and r['job']['noise'] == 0 and r['job']['width'] == 64 and r['job']['depth'] == 1]
    rows = ['| Dataset | Training images | Canonical train % | Validation % | Test % |',
            '| --- | --- | ---: | ---: | ---: |']
    for r in sorted(refs, key=lambda x:(x['dataset'], x['job']['augmentation'])):
        held = f"{100*r['test']['accuracy']:.2f}" if 'test' in r else 'pending'
        rows.append(f"| {r['dataset']} | {r['job']['augmentation']} | {100*r['train_selected_unique']['accuracy']:.2f} | {100*r['validation']['accuracy']:.2f} | {held} |")
    table = '\n'.join(rows)
    text(dest/'README.md', f'''# ELL409: neural correction trees and augmentation

This repository documents two linked assignments: the original neural-tree study and the follow-up on input augmentation, voting ensembles, bias/variance and loss landscapes. It includes unsuccessful attempts as well as positive results.

**Follow-up snapshot: {label}. {completed}/{snapshot['planned_primary_models']} primary configurations have completed. The study is still running; this upload is not a final follow-up report.** Read the dates before comparing snapshots. A configuration is a saved combination of settings; continued epoch budgets are not independent training runs.

## Start here

1. [Experiment journey](docs/EXPERIMENT_JOURNEY.md): what we tried and why.
2. [Results and limitations](docs/RESULTS_AND_LIMITATIONS.md): measured outcomes and the double-descent evidence.
3. [Follow-up methods](docs/FOLLOWUP_METHODS.md): how the four new questions are implemented.
4. [Figure guide](docs/FIGURE_GUIDE.md): which figures answer each question.
5. [Reproduction](docs/REPRODUCE.md): setup, tests, image inference and saved-data plots.

The [previous assignment's PDF](report/ELL409_Previous_Assignment_Report.pdf) is preserved as historical work. It does not report the ongoing augmentation study. The current evidence is under [followup](followup/RESULTS.md), including [all completed measurements](followup/measurements.csv).

## Initial follow-up comparison

{table}

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
''')
    text(dest/'docs/EXPERIMENT_JOURNEY.md', '''# What we tried and why

| Version | Decision and motivation | What it taught us |
| --- | --- | --- |
| V1: shared trained CNN | Compute one useful image representation and share it across small correction networks. This was feasible on a 4 GB RTX 2050. | Excellent training fit, but the strong representation and root classifier explained much of the performance. A successful classifier alone did not prove the tree's value. |
| V2: pixels and selective local CNN nodes | Reduce reliance on one strong shared trunk. Compare tree learning against parameter-matched ordinary classifiers. | The restricted tree was harder to fit. Some ordinary MLP controls were better; those negative results are retained. |
| V3: CNN stopped near 80% | Preserve useful visual features while leaving mistakes for the tree. Freeze its root decisions and increase correction capacity. | Corrections increased fit. A verified two-stage positive-margin parent refit fixed a fitting limitation without relearning the node bases. |
| V4: frozen random CNN | Keep a structured image representation without supervised CNN memorization. | More work fell to the tree. Some double-descent-like hints appeared, but they were not robust across seeds or topology changes. |
| V5: training-label noise | Keep the features and splits fixed, corrupt 15% of training targets, vary capacity and compare root-only controls. | Noisy CIFAR showed a suggestive fall-rise-fall in test error and repeatable improvement among interpolating models. The wide-model effect came mainly from roots, not correction depth. |
| V6: input augmentation | Build a finite image-view bank compatible with hard correction targets; compare against repeated originals and frozen roots. Add grouped bootstrap voting, empirical bias/variance and hard-gate loss slices. | The initial reference models benefit modestly from augmentation on test images. Full width/noise and longer ensemble comparisons remain incomplete at this snapshot. |

These are adaptations and controlled experiments using established methods. We do not claim to have invented CNNs, bagging, double descent or Grad-CAM. The contribution is how the controls expose where learning occurs and where the recursive correction mechanism helps or fails.

Effort is supported by saved code, fixed configurations, negative controls, reproducibility tests, measurements and figures. The number of files or epochs is not itself evidence of scientific success. Version histories reuse datasets and sometimes exactly matching checkpoints; they are not independent replications.
''')
    text(dest/'docs/FOLLOWUP_METHODS.md', '''# Follow-up methods

## 1. Input augmentation, overfitting and capacity

Both datasets use 32 x 32 inputs and the earlier CNN stopped near 80% training accuracy. The CNN and classification head are frozen. Every correction node receives the same 128 normalized CNN features; it cannot read the cached class-margin columns. A CIFAR classifier contains ten one-versus-rest correction trees, while Cats vs Dogs uses one binary tree.

For each original training image, cache its unchanged view and two reflection-padded crop/flip views. Image ID, view number and seed determine the transform. Augmentation happens on pixels before CNN extraction. This finite bank makes zero training-view classification error well defined. Validation/test inputs and labels remain unchanged.

The three reference arms are original only, three repeated original views, and three views including transforms. The repetition control matches image-view exposure, not exact optimizer-update counts: child masks can produce different task sizes. Repeated images are not independent data. Label corruption is a separate target-noise control, not input augmentation.

At each parent, child A addresses false negatives and adds a nonnegative correction; child B addresses false positives and subtracts one. The source implements all four label/prediction cases and excludes don't-care examples from each child's loss. Children feed their recursively corrected hard decisions upward. Nonnegative signed parent coefficients use the verified positive-margin LP if feasible, otherwise a signed Adam fit guarded by training error and loss.

Widths 16/32/64/128/256/512 are compared at fixed depth one. Depth limits 0/1/2/4/8/12 are compared at width 64. The cap is 127 nodes per binary tree. Allocated and active parameters are both recorded, counting the shared CNN once. An increased depth limit need not increase actual model size.

Node optimization uses class-balanced binary cross-entropy and Adam at 0.001. Checkpoint selection minimizes assigned training errors, then unweighted loss. No dropout, smoothing or weight decay is used. Augmentation itself changes and can regularize the fitting problem. It does not guarantee double descent.

## 2. Majority-vote ensembles

Train three complete classifiers on stratified bootstrap draws, keeping all views of a sampled original together and retaining draw multiplicity. Each member votes for one final class. The ten CIFAR one-versus-rest trees constitute one member, not ten votes. Compare one and three members. Ties, which remain possible for multiclass predictions, choose the smallest class index. Log member disagreement and mean member accuracy alongside voting accuracy.

## 3. Samples, epochs, bias and variance

Draw counts are 25%, 50% and 100% of the original training-pool size, with three bootstrap replicates. Unique image counts are lower than draw counts and are logged separately. Per-child epoch budgets are 20, 100 and 400. An epoch means one pass through that node's assigned task, not through the entire CNN.

For validation labels y and member probability vectors p_m, the empirical Brier identity is:

`mean ||p_m-y||^2 = ||mean(p_m)-y||^2 + mean ||p_m-mean(p_m)||^2`.

The first term is a squared-bias proxy against observed labels, not Bayes bias. The second is bootstrap prediction variance, using denominator M. Estimates condition on the same shared CNN previously trained on the full clean pool: varying tree sample size does not measure the sample complexity of retraining the entire image model. Three replicates give limited uncertainty information.

## 4. Loss landscape

The reference models are perturbed in two MLP output-row-normalized random weight directions. CNN weights, roots, biases and parent strengths stay fixed. A 15 x 15 grid measures final classifier training/validation loss on fixed 512-example slices; hard gates are recomputed. The accompanying gate-flip map explains discontinuities.

This is a local slice, not the full landscape. The plotted final loss differs from the collection of masked node objectives used in training. The saved point need not be stationary for the plotted objective; no claim of global convexity or a globally flat minimum follows.

## Evaluation and recovery

CIFAR splits: 45,000 training, 5,000 validation, 10,000 test. Cats splits: 18,608 / 2,326 / 2,326. Reuse saved original manifests and training-only normalization. The three-view banks contain 135,000 CIFAR and 55,824 Cats entries.

Models and repeat selection use training diagnostics; exploration uses validation data. The fixed six-model reference comparison is tested after all six finish. Other test evaluations wait for the primary and training-selected repeat plans. Old test partitions have been reused, so test findings are exploratory. Pending test values must stay blank, not be replaced by validation values.

Matching child tasks resume optimizer and RNG state; changed tasks train afresh. Source/configuration hashes protect each study definition and an OS-held lock prevents duplicate workers. This upload omits the unfinished worker's optimizer checkpoints; resume the original local workspace rather than starting this snapshot as a replacement.
''')
    text(dest/'docs/RESULTS_AND_LIMITATIONS.md', f'''# Results and limitations

Snapshot: **{label}**. Completed primary configurations: **{completed}/{snapshot['planned_primary_models']}**. Figures and tables include only fully saved configurations. Some model losses were still falling at their budget limit; a completed budget does not prove optimization convergence.

## Initial input-augmentation comparison

{table}

Input augmentation improves initial test accuracy by about 0.47 percentage points for Cats and 0.56 points for CIFAR relative to original-only training. This is a single reference comparison, not an uncertainty-tested universal gain. Repeating originals improves fitting but worsens generalization here. Canonical training accuracy differs from accuracy over all augmented training views.

## Can the tree memorize?

The fixed-root depth study reaches 100% original-label training accuracy for Cats without augmentation at depth 12 and for CIFAR at depth 8. CIFAR also fits every finite augmented view at depth 8. These outcomes show correction-tree fitting beyond a frozen imperfect CNN root. Cats augmentation has not interpolated in the completed depth sweep.

## What can we say about double descent?

Interpolation means zero errors on the actual fitting targets and views. The completed depth studies mostly show increasing validation error followed by a plateau. CIFAR depth 12 has the same actual parameter count as depth 8, so it does not test additional effective capacity beyond interpolation. The current width/noise sweep is incomplete; no robust tree-driven double descent is established.

The earlier V5 study is different. With 15% CIFAR training-label noise and 400 epochs, test error at widths 2/8/128/1,024 was 32.60% / 23.43% / 30.21% / 26.29%. This is a double-descent-like shape, but the peak preceded first exact interpolation at width 512. The 1,200-epoch follow-up averaged over three optimizer seeds gave test errors 30.31% / 28.09% / 26.36% at widths 256/512/1,024; all those models interpolated. This establishes repeatable later improvement, not both descents within that follow-up. Wide models had inactive correction children, so that gain was primarily a root-width effect.

The earlier Cats noise follow-up improved at larger widths but did not establish a convincing complete double-descent curve. All these findings use fixed CNNs, one corruption map per dataset and reused test partitions. Label noise is a complementary experiment, not a substitute for the new input-augmentation requirement.

## Ensembles, bias/variance and landscapes

The current [ensemble table](../followup/ensembles.json) compares voting with individual members under matched bootstrap settings; it must not be compared to a different single-model setting as if only voting changed. The [decomposition table](../followup/bias-variance.json) reports a label-based squared-bias proxy and conditional bootstrap variance, not irreducible noise or true Bayes bias.

Landscape maps include gate changes and show only two parameter directions. Observe the measured contours before describing sharpness or flatness. A small number of plotted directions does not establish global properties.

## Verification already performed

The implementation record reports 11 checks passed, a separate 58-configuration small pilot and exact image/cached-prediction agreement on 16 images per dataset for three-member pilot deployments. Pilot accuracy is not full-data performance. That record and the full configurations are in the evidence archive. The package verifier checks current files, hashes and archive contents; it does not certify that every scientific conclusion or queued training job has completed.

## Next experiments should follow evidence

After the current plan, identify a measured interpolation threshold, increase actual active capacity beyond it while keeping the CNN/root fixed, and repeat promising conditions across seeds. Select these runs from training and validation diagnostics. Preserve negative outcomes and report the available evidence rather than requiring a particular curve shape.
''')
    text(dest/'docs/FIGURE_GUIDE.md', '''# Reading the figures

All follow-up figures come from the frozen completed measurements in this snapshot. Lines connect measured configurations; they do not create measurements between them.

| Question | Files under `followup/figures/` | Interpretation |
| --- | --- | --- |
| Does input augmentation help? | `*-reference-comparison.png` | Original versus repeated versus transformed views. Canonical training, validation and reference test accuracy are distinct. |
| Does more tree capacity help after interpolation? | `*-depth-capacity.png` | Fixed root and width, growing depth limit. Dotted line marks first tested exact fit. Repeated x positions can mean the topology stopped growing. |
| Does changing node width produce double descent? | `*-width-capacity.png` | Fixed depth, width-controlled capacity. An unfinished sparse curve cannot establish the entire phenomenon. |
| Does voting help? | `*-ensemble-voting.png` | One versus three complete classifiers under each arm. Some datasets/arms may be missing until enough members finish. |
| How do samples and epoch budgets matter? | `*-bias-variance-samples.png`, `*-bias-variance-epochs.png` | Conditional empirical Brier decomposition. Missing points mean incomplete replicate groups. |
| What does the local loss surface look like? | `*-landscape.png` | Training/validation log1p loss and hard-decision flip fraction on the same perturbation grid. |

Historical figures under `figures/` are explicitly from the previous assignment. `capacity-400.png` shows clean/noisy width controls; `capacity-1200.png` shows the longer three-seed comparison. `feature-strategies.png` summarizes the feature-design progression.

The full historical node, pruning, random-feature and graph-atlas panels are retained in the original project archives. Extract those into a separate folder when reviewing the previous assignment; do not mix their measurement counts with this follow-up.
''')
    text(dest/'docs/REPRODUCE.md', '''# Setup, verification and reproduction

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
''')


def build(workspace, dest):
    workspace = workspace.resolve()
    dest = dest.resolve()
    if dest.exists():
        raise FileExistsError(f'Use a new destination, preserving earlier upload snapshots: {dest}')
    original = workspace/'output/github-upload/ELL409-submission'
    project = workspace/'neural_tree_v6'
    live_run = project/'runs/augmentation-study'
    live_output = project/'results/augmentation-study'
    records = read(live_output/'measurements.json')
    progress = read(live_run/'progress.json')
    definition = read(live_run/'definition.json')
    for name, expected in definition['source_hashes'].items():
        assert sha(project/name) == expected, f'Live source changed: {name}'
    dest.mkdir(parents=True)
    archive_dir = dest/'archives'
    archive_dir.mkdir()
    snapshot = {'captured_ist': datetime.now(IST).isoformat(timespec='seconds'),
                'completed_primary_models': len(records),
                'planned_primary_models': progress['planned_primary_models'],
                'study_complete': progress['status'] == 'complete',
                'worker_progress_at_capture': progress,
                'scope': 'Completed configurations only; live worker and unfinished checkpoints are not modified.'}
    print(f'Capturing {len(records)} completed models; live training is untouched.', flush=True)

    # Core source is copied from the exact current workspace, not an older ZIP.
    sources = list((workspace/'neural_tree/ell409_tree').glob('*.py'))
    sources += list((workspace/'neural_tree_v2/v2_tree').glob('*.py'))
    for name in ['neural_tree_v3/study.py', 'neural_tree_v3/width32_followup.py',
                 'neural_tree_v4/random_cnn_study.py', 'neural_tree_v5/noise_study.py',
                 'neural_tree_v5/noise_followup.py', 'neural_tree_v5/deploy_noise.py',
                 'neural_tree/pyproject.toml', 'neural_tree_v2/pyproject.toml',
                 'neural_tree_v3/config.json', 'neural_tree_v4/config.json',
                 'neural_tree_v5/config.json', 'neural_tree_v5/followup-config.json']:
        sources.append(workspace/name)
    sources += list(project.glob('*.py'))
    sources += [project/p for p in ['config.json', 'pilot.json', 'requirements.txt',
                                   'requirements-lock.txt', 'README.md', 'tests/test_followup.py']]
    for source in sources:
        copy(source, dest/source.relative_to(workspace))
    with ZipFile(original/'project-02.zip') as z:
        (dest/'requirements.txt').write_bytes(z.read('requirements.txt'))
    for name in ['capacity-400.png', 'capacity-1200.png', 'feature-strategies.png']:
        path = dest/'figures'/name
        path.parent.mkdir(exist_ok=True)
        for part in ['project-01.zip', 'project-02.zip']:
            with ZipFile(original/part) as z:
                if 'figures/'+name in z.namelist():
                    path.write_bytes(z.read('figures/'+name))
                    break
        else:
            raise FileNotFoundError(f'Historical figure missing from both archives: {name}')
    for name in ['project-01.zip', 'project-02.zip']:
        copy(original/name, archive_dir/name)
    copy(original/'SUBMISSION_INDEX.json', archive_dir/'PREVIOUS_SUBMISSION_INDEX.json')
    copy(original/'ELL409_Final_Report.pdf', dest/'report/ELL409_Previous_Assignment_Report.pdf')

    # Build immutable plotting inputs outside both the live run and upload folder.
    stage = dest.parent/(dest.name+'-snapshot-inputs')
    if stage.exists():
        raise FileExistsError(f'Snapshot staging already exists: {stage}')
    stage_run = stage/'neural_tree_v6/runs/augmentation-study'
    stage_out = stage/'followup'
    stage_run.mkdir(parents=True)
    stage_out.mkdir(parents=True)
    for name in ['definition.json', 'primary-plan.json']:
        copy(live_run/name, stage_run/name)
    write(stage_run/'captured-progress.json', progress)
    copy(project/'results/implementation-verification.json', stage_run/'implementation-verification.json')
    for r in records:
        folder = Path(r['model_path']).parent
        write(stage_run/folder/'metrics.json', r)
        copy(live_run/r['predictions_path'], stage_run/r['predictions_path'])
    for source in (live_output/'landscapes').rglob('*'):
        if source.is_file():
            copy(source, stage_out/source.relative_to(live_output))
    for source in (live_output/'models').glob('*.pt'):
        copy(source, stage_out/'models'/source.name)

    # Only the snapshot is summarized/rewritten by this process.
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MPLCONFIGDIR=str(stage/'matplotlib-cache'))
    command = [sys.executable, '-B', str(project/'plots.py'), '--run-dir', str(stage_run), '--output', str(stage_out)]
    print('Regenerating figures from frozen completed evidence, without training.', flush=True)
    subprocess.run(command, cwd=workspace, env=env, check=True)
    assert {r['job_id'] for r in read(stage_out/'measurements.json')} == {r['job_id'] for r in records}
    for source in stage_out.iterdir():
        if source.is_file():
            copy(source, dest/'followup'/source.name)
    for source in (stage_out/'figures').glob('*.png'):
        copy(source, dest/'followup/figures'/source.name)
    write(dest/'followup/SNAPSHOT.json', snapshot)
    model_entries = archive_parts([(p, p.relative_to(stage).as_posix())
                                  for p in sorted((stage_out/'models').glob('*.pt'))], archive_dir, 'followup-models')
    raw = [(p, p.relative_to(stage).as_posix()) for p in sorted(stage_run.rglob('*')) if p.is_file()]
    raw += [(p, p.relative_to(stage).as_posix()) for p in sorted((stage_out/'landscapes').rglob('*')) if p.is_file()]
    evidence_entries = archive_parts(raw, archive_dir, 'followup-evidence')
    make_documents(dest, snapshot, records)
    text(dest/'tools/verify_repository.py', VERIFY)
    copy(Path(__file__), dest/'tools/refresh_repository.py')
    text(dest/'.gitignore', '''.venv/
__pycache__/
*.pyc
.pytest_cache/
historical-restored/
neural_tree/data/
neural_tree/runs/
neural_tree_v2/runs/
neural_tree_v3/runs/
neural_tree_v4/runs/
neural_tree_v5/runs/
neural_tree_v6/runs/
followup/models/
followup/landscapes/
''')
    files = sorted(p for p in dest.rglob('*') if p.is_file())
    assert len(files)+1 < 100, f'Too many upload files: {len(files)+1}'
    assert all(p.stat().st_size < 25*1024*1024 for p in files)
    manifest = {'format': 'ELL409-expanded-repository-v1', 'repository_url': 'https://github.com/Harshal-Narsaria/ELL409',
                'snapshot': snapshot, 'physical_upload_files': len(files)+1,
                'source_provenance': [{'path': s.relative_to(workspace).as_posix(), 'sha256': sha(s)} for s in sources],
                'legacy_archives': [{'path': 'archives/'+name, 'sha256': sha(archive_dir/name)}
                                    for name in ['project-01.zip', 'project-02.zip']],
                'followup_archive_entries': model_entries+evidence_entries,
                'files': [{'path': p.relative_to(dest).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]}
    write(dest/'PACKAGE_INDEX.json', manifest)
    subprocess.run([sys.executable, '-B', str(dest/'tools/verify_repository.py')], check=True)
    for name, expected in definition['source_hashes'].items():
        assert sha(project/name) == expected
    print(json.dumps({'destination': str(dest), 'upload_files': len(files)+1,
                      'snapshot_models': len(records), 'figures': len(list(dest.rglob('*.png'))),
                      'source_files': len(list(dest.rglob('*.py'))),
                      'total_megabytes': round(sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())/1048576, 2)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    destination = args.destination or args.workspace/'output/github-upload/ELL409-expanded'
    build(args.workspace, destination)
