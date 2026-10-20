# What we tried and why

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
