# V4: frozen random-CNN features

An **untrained** CNN (16/32/64 channels, 2×2 pooling → 256 features) supplies the features, and a trained width-8 root plus correction trees do all the supervised learning. Corrections take training accuracy to 100% on both datasets, but test accuracy is low.

| File | Contents |
| --- | --- |
| `random_cnn_study.py` | `RandomCNN`, the tree trainer, the width and depth studies |
| `queue_random_cnn.py`, `plot_test_accuracy.py` | Background runner, plotting |
| `config.json` | Settings |
