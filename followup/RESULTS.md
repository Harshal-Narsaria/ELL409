# Augmentation follow-up: completed evidence

Completed saved model configurations: 85.

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
| cats_vs_dogs | augmented | 0% | 64 | 1 | 81.99 | 81.30 | 81.64 |
| cats_vs_dogs | augmented | 0% | 64 | 2 | 85.40 | 81.30 | withheld |
| cats_vs_dogs | augmented | 0% | 64 | 4 | 87.29 | 80.74 | withheld |
| cats_vs_dogs | augmented | 0% | 64 | 8 | 88.17 | 80.65 | withheld |
| cats_vs_dogs | augmented | 0% | 64 | 12 | 88.17 | 80.65 | withheld |
| cats_vs_dogs | augmented | 0% | 512 | 1 | 89.37 | 76.10 | withheld |
| cats_vs_dogs | none | 0% | 64 | 1 | 84.25 | 81.47 | 81.17 |
| cats_vs_dogs | none | 0% | 64 | 2 | 88.41 | 81.04 | withheld |
| cats_vs_dogs | none | 0% | 64 | 4 | 91.94 | 80.18 | withheld |
| cats_vs_dogs | none | 0% | 64 | 8 | 97.61 | 78.46 | withheld |
| cats_vs_dogs | none | 0% | 64 | 12 | 100.00 | 76.53 | withheld |
| cats_vs_dogs | none | 0% | 512 | 1 | 94.21 | 77.82 | withheld |
| cats_vs_dogs | repeat | 0% | 64 | 1 | 86.50 | 80.91 | 80.87 |
| cifar10 | augmented | 0% | 64 | 1 | 88.33 | 77.42 | 76.69 |
| cifar10 | augmented | 0% | 64 | 2 | 93.95 | 76.32 | withheld |
| cifar10 | augmented | 0% | 64 | 4 | 96.85 | 74.90 | withheld |
| cifar10 | augmented | 0% | 64 | 8 | 100.00 | 72.96 | withheld |
| cifar10 | augmented | 0% | 64 | 12 | 100.00 | 72.96 | withheld |
| cifar10 | augmented | 0% | 512 | 1 | 100.00 | 74.84 | withheld |
| cifar10 | none | 0% | 64 | 1 | 94.46 | 76.50 | 76.13 |
| cifar10 | none | 0% | 64 | 2 | 98.28 | 76.26 | withheld |
| cifar10 | none | 0% | 64 | 4 | 99.74 | 73.82 | withheld |
| cifar10 | none | 0% | 64 | 8 | 100.00 | 73.64 | withheld |
| cifar10 | none | 0% | 64 | 12 | 100.00 | 73.64 | withheld |
| cifar10 | none | 0% | 512 | 1 | 100.00 | 75.38 | withheld |
| cifar10 | repeat | 0% | 64 | 1 | 99.26 | 74.78 | 74.44 |
