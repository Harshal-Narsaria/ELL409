# Reproducing the results

## 1. Environment

The code was run with Python 3.12, PyTorch 2.14 (CUDA 12.6), NumPy 2.3, SciPy 1.18 and Matplotlib 3.11 on an NVIDIA RTX 2050 (4 GB).

```bash
pip install torch torchvision   # pick the CPU or CUDA build from pytorch.org
pip install -r requirements.txt
```

Preparing Cats vs Dogs also needs `tensorflow-datasets` (see `neural_tree/pyproject.toml`, the `cats` extra).

## 2. Tests (no data needed, about 3 minutes)

```bash
cd neural_tree_v6 && python -m pytest tests
```

Every version folder has its own `tests/` directory and can be run the same way. They use small synthetic inputs.

## 3. Try the saved models (no data needed)

```bash
python neural_tree_v6/deploy.py models/assignment2/cifar10-augmented-vote-3.pt examples/cifar10_test_0_cat.png
```

Any RGB image works; it is resized to 32×32. The models in `models/assignment2/` are:

| File | Model |
| --- | --- |
| `<dataset>-<arm>-single.pt` | one width-64, depth-1 reference tree trained on `none`, `repeat` or `augmented` images |
| `<dataset>-<arm>-vote-3.pt` | three bootstrap trees voting (400 epochs per child) |

Each file contains the frozen CNN as well, so it goes straight from image to prediction.

## 4. Rerunning the experiments

Run all commands from the repository root. The datasets, cached features and checkpoints are not in the repository; the scripts create them under `neural_tree*/data` and `neural_tree*/runs`.

```bash
# (a) Download data and build the fixed train/validation/test splits.
cd neural_tree
python -m ell409_tree prepare cifar10 --data-dir data
python -m ell409_tree prepare cats_vs_dogs --data-dir data
cd ..

# (b) Train the CNN until 80% training accuracy, freeze it and cache its features.
#     This also runs the V3 depth study.
python neural_tree_v3/study.py --device cuda

# (b2) V5 label-noise study. It writes the fixed 15% noise map that V6's noise control reuses.
python neural_tree_v5/noise_study.py --device cuda

# (c) Assignment 2 main study: 166 configurations, resumable; rerun the same command to continue.
python neural_tree_v6/run.py --hours 10

# (d) Assignment 2 noisy 4,000-image subset width study (report section 7). Needs (c).
python neural_tree_v6/subset_capacity_study.py --no-time-limit
```

Assignment 1 (V1) runs with `python -m ell409_tree run --manifest data/cifar10/manifest.json --config configs/cifar10.json --run-dir runs/cifar10` from inside `neural_tree/`. Use `python -m ell409_tree --help` for the other commands.

Settings for each study are in the `config.json` files next to the scripts. The master seed is 409 and repeats use 410 and 411. Every run checks a hash of its configuration and inputs, so it refuses to resume after something has changed.
