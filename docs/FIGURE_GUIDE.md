# Reading the figures

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
