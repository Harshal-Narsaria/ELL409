# Follow-up methods

## 1. Input augmentation, overfitting and capacity

Both datasets use 32 x 32 inputs and the earlier CNN stopped near 80% training accuracy. The CNN and classification head are frozen. Every correction node receives the same 128 normalized CNN features; it cannot read the cached class-margin columns. A CIFAR classifier contains ten one-versus-rest correction trees, while Cats vs Dogs uses one binary tree.

For each original training image, cache its unchanged view and two reflection-padded crop/flip views. Image ID, view number and seed determine the transform. Augmentation happens on pixels before CNN extraction. This finite bank makes zero training-view classification error well defined. Validation/test inputs and labels remain unchanged.

The three reference arms are original only, three repeated original views, and three views including transforms. The repetition control matches image-view exposure, not exact optimizer-update counts: child masks can produce different task sizes. Repeated images are not independent data. Label corruption is a separate target-noise control, not input augmentation.

At each parent, child A addresses false negatives and adds a nonnegative correction; child B addresses false positives and subtracts one. The source implements all four label/prediction cases and excludes don't-care examples from each child's loss. Children feed their recursively corrected hard decisions upward. Nonnegative signed parent coefficients use the verified positive-margin LP if feasible, otherwise a signed Adam fit guarded by training error and loss.

Widths 16/32/64/128/256/512 are compared at fixed depth one. Depth limits 0/1/2/4/8/12 are compared at width 64. The cap is 127 nodes per binary tree. Allocated and active parameters are both recorded, counting the shared CNN once. An increased depth limit need not increase actual model size.

Node optimization uses class-balanced binary cross-entropy and Adam at 0.001. Checkpoint selection minimizes assigned training errors, then unweighted loss. No dropout, smoothing or weight decay is used. Augmentation itself changes and can regularize the fitting problem. It does not guarantee double descent.

## 2. Majority-vote ensembles

Train three complete classifiers on stratified bootstrap draws, keeping all views of a sampled original together and retaining draw multiplicity. Each member votes for one final class. The ten CIFAR one-versus-rest trees constitute one member, not ten votes. Compare one and three members. Ties, which remain possible for multiclass predictions, choose the smallest class index. Log member disagreement and mean member accuracy alongside voting accuracy.

## 3. Samples, epochs, bias and variance

Draw counts are 25%, 50% and 100% of the original training-pool size, with three bootstrap replicates. Unique image counts are lower than draw counts and are logged separately. Per-child epoch budgets are 20, 100 and 400. An epoch means one pass through that node's assigned task, not through the entire CNN.

For validation labels y and member probability vectors p_m, the empirical Brier identity is:

`mean ||p_m-y||^2 = ||mean(p_m)-y||^2 + mean ||p_m-mean(p_m)||^2`.

The first term is a squared-bias proxy against observed labels, not Bayes bias. The second is bootstrap prediction variance, using denominator M. Estimates condition on the same shared CNN previously trained on the full clean pool: varying tree sample size does not measure the sample complexity of retraining the entire image model. Three replicates give limited uncertainty information.

## 4. Loss landscape

The reference models are perturbed in two MLP output-row-normalized random weight directions. CNN weights, roots, biases and parent strengths stay fixed. A 15 x 15 grid measures final classifier training/validation loss on fixed 512-example slices; hard gates are recomputed. The accompanying gate-flip map explains discontinuities.

This is a local slice, not the full landscape. The plotted final loss differs from the collection of masked node objectives used in training. The saved point need not be stationary for the plotted objective; no claim of global convexity or a globally flat minimum follows.

## Evaluation and recovery

CIFAR splits: 45,000 training, 5,000 validation, 10,000 test. Cats splits: 18,608 / 2,326 / 2,326. Reuse saved original manifests and training-only normalization. The three-view banks contain 135,000 CIFAR and 55,824 Cats entries.

Models and repeat selection use training diagnostics; exploration uses validation data. The fixed six-model reference comparison is tested after all six finish. Other test evaluations wait for the primary and training-selected repeat plans. Old test partitions have been reused, so test findings are exploratory. Pending test values must stay blank, not be replaced by validation values.

Matching child tasks resume optimizer and RNG state; changed tasks train afresh. Source/configuration hashes protect each study definition and an OS-held lock prevents duplicate workers. This upload omits the unfinished worker's optimizer checkpoints; resume the original local workspace rather than starting this snapshot as a replacement.
