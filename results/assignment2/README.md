# Assignment 2 measurements

| File | Contents |
| --- | --- |
| `measurements.csv` | One row per trained configuration (210): dataset, arm, noise, width, depth, epochs, sample fraction, seed, parameters, train/validation/test accuracy |
| `RESULTS.md` | Auto-generated summary tables |
| `ensembles.json` | 1-member vs 3-member majority-vote accuracy, member disagreement |
| `bias-variance.json` | Brier decomposition (bias proxy, variance) for each sample fraction × epoch group |
| `reference-test.json` | One-time test evaluation of the six reference models |
| `ensemble-tests/` | One-time test evaluation of each voting ensemble |
| `noisy-subset/measurements.json` | Noisy 4,000-image subset width study |
| `view-voting-and-correction-diagnostics.json` | 8-view test-time voting and fixed/spoiled root-prediction counts (validation) |
