# Augmentation follow-up: completed evidence

Completed saved model configurations: 210.

These figures use completed configurations only. Canonical images, augmented views and bootstrap image draws are counted separately.

## Evidence boundaries

Conditional on the original shared CNN trained on the full clean training pool. Original sample counts refer to correction-tree training draws; augmented views are not independent originals. Empirical squared-bias proxy uses observed held-out one-hot labels; irreducible noise is not identified.

Reused original test partitions, exploratory. A fixed six-model reference comparison is evaluated once all six predeclared reference models finish. Remaining tests wait for the full primary and training-selected repeat plans. No test score changes the plan or selects models.

Training curves locate interpolation. Held-out curves determine whether double descent is present; it is not guaranteed.

Epochs in these plots mean passes through each child task. Frozen roots do not train. The depth-one epoch study has fixed child tasks and resumes them at common epoch budgets.

Loss slices perturb MLP weights, keep parent strengths fixed, and recompute hard gates. They cannot establish a smooth Hessian or a global minimum.

## Latest complete full-data models

| Dataset | Augmentation | Noise | Width | Depth | Train views % | Validation % | Test % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cats_vs_dogs | augmented | 0% | 16 | 1 | 81.90 | 81.73 | 82.46 |
| cats_vs_dogs | augmented | 15% | 32 | 1 | 72.47 | 81.81 | 82.07 |
| cats_vs_dogs | augmented | 0% | 32 | 1 | 81.75 | 81.60 | 82.07 |
| cats_vs_dogs | augmented | 0% | 64 | 0 | 81.74 | 81.60 | 82.07 |
| cats_vs_dogs | augmented | 0% | 64 | 1 | 81.99 | 81.30 | 81.64 |
| cats_vs_dogs | augmented | 0% | 64 | 2 | 85.40 | 81.30 | 81.17 |
| cats_vs_dogs | augmented | 0% | 64 | 4 | 87.29 | 80.74 | 80.95 |
| cats_vs_dogs | augmented | 0% | 64 | 8 | 88.17 | 80.65 | 81.26 |
| cats_vs_dogs | augmented | 0% | 64 | 12 | 88.17 | 80.65 | 81.26 |
| cats_vs_dogs | augmented | 15% | 128 | 1 | 74.43 | 80.65 | 80.87 |
| cats_vs_dogs | augmented | 0% | 128 | 1 | 82.60 | 81.30 | 80.95 |
| cats_vs_dogs | augmented | 0% | 256 | 1 | 84.91 | 78.80 | 78.63 |
| cats_vs_dogs | augmented | 0% | 512 | 1 | 89.37 | 76.10 | 77.17 |
| cats_vs_dogs | augmented | 15% | 512 | 1 | 84.50 | 71.93 | 71.71 |
| cats_vs_dogs | augmented | 0% | 16 | 1 | 81.83 | 81.47 | 82.42 |
| cats_vs_dogs | augmented | 15% | 32 | 1 | 72.26 | 81.64 | 81.81 |
| cats_vs_dogs | augmented | 15% | 128 | 1 | 74.27 | 80.35 | 81.08 |
| cats_vs_dogs | augmented | 0% | 128 | 1 | 82.68 | 81.47 | 80.83 |
| cats_vs_dogs | augmented | 15% | 512 | 1 | 85.30 | 71.84 | 71.41 |
| cats_vs_dogs | augmented | 0% | 512 | 1 | 89.82 | 77.43 | 76.83 |
| cats_vs_dogs | augmented | 0% | 16 | 1 | 81.77 | 81.73 | 82.20 |
| cats_vs_dogs | augmented | 15% | 32 | 1 | 72.51 | 80.91 | 81.08 |
| cats_vs_dogs | augmented | 15% | 128 | 1 | 74.14 | 80.48 | 81.17 |
| cats_vs_dogs | augmented | 0% | 128 | 1 | 82.26 | 81.13 | 81.60 |
| cats_vs_dogs | augmented | 0% | 512 | 1 | 91.22 | 77.09 | 77.99 |
| cats_vs_dogs | augmented | 15% | 512 | 1 | 85.10 | 72.61 | 72.44 |
| cats_vs_dogs | none | 0% | 16 | 1 | 83.78 | 81.81 | 82.16 |
| cats_vs_dogs | none | 15% | 32 | 1 | 74.07 | 81.43 | 81.64 |
| cats_vs_dogs | none | 0% | 32 | 1 | 83.73 | 81.60 | 82.24 |
| cats_vs_dogs | none | 0% | 64 | 0 | 83.68 | 81.60 | 82.07 |
| cats_vs_dogs | none | 0% | 64 | 1 | 84.25 | 81.47 | 81.17 |
| cats_vs_dogs | none | 0% | 64 | 2 | 88.41 | 81.04 | 80.74 |
| cats_vs_dogs | none | 0% | 64 | 4 | 91.94 | 80.18 | 80.09 |
| cats_vs_dogs | none | 0% | 64 | 8 | 97.61 | 78.46 | 78.20 |
| cats_vs_dogs | none | 0% | 64 | 12 | 100.00 | 76.53 | 75.88 |
| cats_vs_dogs | none | 15% | 128 | 1 | 77.58 | 75.49 | 74.94 |
| cats_vs_dogs | none | 0% | 128 | 1 | 85.20 | 80.95 | 81.21 |
| cats_vs_dogs | none | 0% | 256 | 1 | 88.84 | 78.80 | 77.34 |
| cats_vs_dogs | none | 15% | 512 | 1 | 89.49 | 72.53 | 72.10 |
| cats_vs_dogs | none | 0% | 512 | 1 | 94.21 | 77.82 | 78.42 |
| cats_vs_dogs | none | 0% | 16 | 1 | 83.77 | 81.94 | 82.42 |
| cats_vs_dogs | none | 15% | 32 | 1 | 73.82 | 81.38 | 81.43 |
| cats_vs_dogs | none | 15% | 128 | 1 | 76.48 | 80.27 | 80.05 |
| cats_vs_dogs | none | 0% | 128 | 1 | 84.99 | 81.04 | 81.00 |
| cats_vs_dogs | none | 15% | 512 | 1 | 89.79 | 71.50 | 70.34 |
| cats_vs_dogs | none | 0% | 512 | 1 | 94.55 | 78.16 | 77.99 |
| cats_vs_dogs | none | 0% | 16 | 1 | 83.80 | 81.73 | 81.81 |
| cats_vs_dogs | none | 15% | 32 | 1 | 74.06 | 81.21 | 80.74 |
| cats_vs_dogs | none | 0% | 128 | 1 | 84.95 | 80.91 | 81.17 |
| cats_vs_dogs | none | 15% | 128 | 1 | 76.66 | 77.09 | 75.97 |
| cats_vs_dogs | none | 15% | 512 | 1 | 89.51 | 71.28 | 71.24 |
| cats_vs_dogs | none | 0% | 512 | 1 | 93.45 | 78.42 | 77.77 |
| cats_vs_dogs | repeat | 0% | 64 | 1 | 86.50 | 80.91 | 80.87 |
| cifar10 | augmented | 0% | 16 | 1 | 79.91 | 77.96 | 77.88 |
| cifar10 | augmented | 15% | 32 | 1 | 68.87 | 76.78 | 76.29 |
| cifar10 | augmented | 0% | 32 | 1 | 82.45 | 78.06 | 77.53 |
| cifar10 | augmented | 0% | 64 | 0 | 75.07 | 75.76 | 75.70 |
| cifar10 | augmented | 0% | 64 | 1 | 88.33 | 77.42 | 76.69 |
| cifar10 | augmented | 0% | 64 | 2 | 93.95 | 76.32 | 76.09 |
| cifar10 | augmented | 0% | 64 | 4 | 96.85 | 74.90 | 74.11 |
| cifar10 | augmented | 0% | 64 | 8 | 100.00 | 72.96 | 72.03 |
| cifar10 | augmented | 0% | 64 | 12 | 100.00 | 72.96 | 72.03 |
| cifar10 | augmented | 15% | 128 | 1 | 78.12 | 72.80 | 72.86 |
| cifar10 | augmented | 0% | 128 | 1 | 95.87 | 75.68 | 75.27 |
| cifar10 | augmented | 0% | 256 | 1 | 99.13 | 74.64 | 74.75 |
| cifar10 | augmented | 15% | 512 | 1 | 95.92 | 67.32 | 66.48 |
| cifar10 | augmented | 0% | 512 | 1 | 100.00 | 74.84 | 74.27 |
| cifar10 | augmented | 15% | 32 | 1 | 68.93 | 76.20 | 76.56 |
| cifar10 | augmented | 15% | 128 | 1 | 77.94 | 72.94 | 73.46 |
| cifar10 | augmented | 0% | 256 | 1 | 99.19 | 74.62 | 74.84 |
| cifar10 | augmented | 15% | 512 | 1 | 96.49 | 66.70 | 66.20 |
| cifar10 | augmented | 0% | 512 | 1 | 100.00 | 74.76 | 74.14 |
| cifar10 | augmented | 15% | 32 | 1 | 69.00 | 76.50 | 76.66 |
| cifar10 | augmented | 15% | 128 | 1 | 77.93 | 73.42 | 72.34 |
| cifar10 | augmented | 0% | 256 | 1 | 99.19 | 74.58 | 74.57 |
| cifar10 | augmented | 0% | 512 | 1 | 100.00 | 73.98 | 74.55 |
| cifar10 | augmented | 15% | 512 | 1 | 95.90 | 68.52 | 67.59 |
| cifar10 | none | 0% | 16 | 1 | 84.80 | 78.48 | 77.85 |
| cifar10 | none | 0% | 32 | 1 | 88.49 | 77.54 | 77.24 |
| cifar10 | none | 15% | 32 | 1 | 72.80 | 76.84 | 77.30 |
| cifar10 | none | 0% | 64 | 0 | 80.14 | 75.76 | 75.70 |
| cifar10 | none | 0% | 64 | 1 | 94.46 | 76.50 | 76.13 |
| cifar10 | none | 0% | 64 | 2 | 98.28 | 76.26 | 75.88 |
| cifar10 | none | 0% | 64 | 4 | 99.74 | 73.82 | 73.46 |
| cifar10 | none | 0% | 64 | 8 | 100.00 | 73.64 | 73.58 |
| cifar10 | none | 0% | 64 | 12 | 100.00 | 73.64 | 73.58 |
| cifar10 | none | 0% | 128 | 1 | 98.67 | 75.94 | 75.58 |
| cifar10 | none | 15% | 128 | 1 | 85.46 | 70.10 | 70.46 |
| cifar10 | none | 0% | 256 | 1 | 99.22 | 75.12 | 75.36 |
| cifar10 | none | 15% | 512 | 1 | 98.29 | 65.10 | 65.70 |
| cifar10 | none | 0% | 512 | 1 | 100.00 | 75.38 | 75.07 |
| cifar10 | none | 15% | 32 | 1 | 73.07 | 77.24 | 77.10 |
| cifar10 | none | 15% | 128 | 1 | 85.48 | 70.68 | 70.41 |
| cifar10 | none | 0% | 256 | 1 | 99.19 | 75.52 | 75.57 |
| cifar10 | none | 15% | 512 | 1 | 97.66 | 66.84 | 66.08 |
| cifar10 | none | 0% | 512 | 1 | 100.00 | 74.94 | 75.37 |
| cifar10 | none | 15% | 32 | 1 | 73.15 | 77.00 | 76.87 |
| cifar10 | none | 15% | 128 | 1 | 85.56 | 70.94 | 71.03 |
| cifar10 | none | 0% | 256 | 1 | 99.21 | 75.56 | 75.66 |
| cifar10 | none | 0% | 512 | 1 | 100.00 | 75.10 | 75.53 |
| cifar10 | none | 15% | 512 | 1 | 97.72 | 65.24 | 66.21 |
| cifar10 | repeat | 0% | 64 | 1 | 99.26 | 74.78 | 74.44 |
