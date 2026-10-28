# V3: CNN stopped at 80% as a frozen root

The CNN is trained only until its first epoch above 80% training accuracy, then frozen. Its class margin becomes the root score, and correction trees of increasing depth learn the remaining errors.

| File | Contents |
| --- | --- |
| `study.py` | `root_scores`, `FeatureNode`, `FeatureForest` (**used by the final V6 model**), `train_to_target` (80% CNN), feature extraction, `ProgressiveTrainer` |
| `width32_followup.py` | `classification_refit`: the repaired parent LP (maximise margin ≤ 1, then minimise α+β), **used by V5 and V6** |
| `config.json`, `width32-followup.json` | Settings |
| `queue-study.py` | Runs the study in the background |

Run: `python study.py --device cuda`. This also writes the CNN80 features and checkpoint that V5 and V6 read.
