# V2: pixel and local-CNN nodes

The nodes read raw 32×32 pixels (`PixelScore`), or train their own small CNN at selected nodes (`ConvScore`). This tests whether the tree can learn without a strong shared CNN. The results are compared with parameter-matched ordinary MLPs and CNNs.

| File | Contents |
| --- | --- |
| `v2_tree/models.py` | `correction_targets`, `ImageNode` (corrected score), `ImageForest` |
| `v2_tree/train.py` | `fit_node`, `fit_correction`, `ImageTreeTrainer.grow` |
| `v2_tree/study.py` | Study runner, pruning, result export |
| `v2_tree/common.py` | Checkpointing, hashing, RNG save/restore and run locks (reused by V3 to V6) |
| `configs/` | Study settings |

Run with `python -m v2_tree --help` from this folder.
