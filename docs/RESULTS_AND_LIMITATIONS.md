# Results and limitations

Snapshot: **2026-10-20T11:21:48+05:30**. Completed primary configurations: **85/166**. Figures and tables include only fully saved configurations. Some model losses were still falling at their budget limit; a completed budget does not prove optimization convergence.

## Initial input-augmentation comparison

| Dataset | Training images | Canonical train % | Validation % | Test % |
| --- | --- | ---: | ---: | ---: |
| cats_vs_dogs | augmented | 83.74 | 81.30 | 81.64 |
| cats_vs_dogs | none | 84.25 | 81.47 | 81.17 |
| cats_vs_dogs | repeat | 86.50 | 80.91 | 80.87 |
| cifar10 | augmented | 90.67 | 77.42 | 76.69 |
| cifar10 | none | 94.46 | 76.50 | 76.13 |
| cifar10 | repeat | 99.26 | 74.78 | 74.44 |

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
