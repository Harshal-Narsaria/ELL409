# V5: 15% training-label noise

This version uses the CNN80 features from V3 and corrupts 15% of training labels to a different class (the same map at every width). It then sweeps node width from 2 to 1,024 and compares against root-only controls. Validation and test labels stay clean.

| File | Contents |
| --- | --- |
| `noise_study.py` | `corrupt_labels` (fixed noise map), `FixedTrainer` (depth-1 tree), the width study, test finalisation |
| `noise_followup.py`, `bounded_followup.py`, `finish_followup.py` | 1,200-epoch, three-seed follow-up |
| `root_gap.py`, `evaluate_root_gap_at_cutoff.py` | Root-only versus tree comparison |
| `deploy_noise.py` | Image → prediction for saved models |
| `*.json` | Settings for each sub-study |

V6 reuses the noise map written by `noise_study.py` for its own label-noise control.
