# ELL409: Tree-Type Neural Networks (Neural Error-Correction Trees)

**Harshal Narsaria · ELL409 Machine Learning and Optimization · IIT Delhi · 2026**

This repository holds the code, results and reports for two linked assignments:

1. **Tree type network revisited.** Build a tree-type neural network whose nodes are small neural networks, overfit CIFAR-10 and Cats vs Dogs, look for double descent, interpret the nodes and shrink the model.
2. **Tree type NN: exploring double descent and other aspects.** Retrain the tree with input augmentation, use multiple trees with majority voting, study bias/variance against samples and epochs, and comment on the loss landscape ([brief](reports/assignment2_brief.png)).

## Start here

| Read | What it gives you |
| --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | How the model works: correction targets, tree growth, parent fitting, augmentation, voting, bias/variance and landscape. Diagrams included. |
| [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) | Where each idea is implemented, with links to the exact functions and a 30-minute reading order. |
| [docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md) | The six versions (V1 → V6): what changed each time, why, and what we learned. |
| [docs/RESULTS.md](docs/RESULTS.md) | Main tables and figures for both assignments. |
| [reports/](reports) | The submitted PDF reports. |

## The model in brief

Each node is a small MLP that gives a score; the node predicts class 1 when the score is ≥ 0. When a node makes mistakes, it gets two children. **Child A** learns to recognise the node's false negatives and **child B** its false positives. The parent corrects its own score using their hard decisions:

```text
score(x) = base(x) + α·[A(x) ≥ 0] − β·[B(x) ≥ 0] + offset,     α, β ≥ 0
```

Children can be corrected by their own children, so the tree grows where errors remain. CIFAR-10 uses ten one-vs-rest trees. In the final design the root is a **frozen CNN stopped at 80% training accuracy**, so every improvement over that CNN comes from the correction tree.

## Key findings

| Question | Finding |
| --- | --- |
| Can the tree overfit? | Yes. 99.9933% (CIFAR-10) and 100% (Cats vs Dogs) training accuracy in V1; 100% on top of a frozen imperfect CNN in V6. |
| Does augmentation help? | Slightly. Best test accuracy on both datasets: CIFAR-10 76.13 → 76.69%, Cats 81.17 → 81.64%. Repeating the same images instead overfits. |
| Double descent? | Not on the full data within the tested widths. **Yes** on a noisy (15%) 4,000-image CIFAR subset: test error peaks near interpolation, then falls by 4 to 7 points at larger widths. |
| Majority voting? | Three bootstrap trees beat a single tree in all 12 settings, by up to +1.5 points. |
| Bias / variance? | Variance grows sharply with epochs and shrinks with more samples at long training. |
| Loss landscape? | Hard child decisions flip under small weight changes (up to 21% of node decisions), which makes the landscape step-like rather than smooth. |

## Repository layout

```text
docs/              design, code walkthrough, experiment history, results, reproduction
reports/           Assignment 1 and Assignment 2 PDF reports, assignment brief
figures/           figures used in the reports (assignment1/, assignment2/)
results/           Assignment 2 measurements: all 210 configurations, ensembles, bias/variance, test results
models/            ready-to-run Assignment 2 models (single trees and 3-tree voting ensembles)
examples/          one CIFAR-10 test image to try the models on

neural_tree/       V1: shared CNN + correction tree package (Assignment 1 main model)
neural_tree_v2/    V2: pixel / local-CNN nodes with parameter-matched controls
neural_tree_v3/    V3: CNN stopped at 80% as a frozen root; repaired parent LP
neural_tree_v4/    V4: frozen random-CNN features
neural_tree_v5/    V5: 15% label-noise width study
neural_tree_v6/    V6: Assignment 2 (augmentation, voting, bias/variance, loss landscape)
```

Later versions import code from earlier ones by folder name, so the folders keep their original names. Each folder has its own short README.

## Quick start

```bash
pip install -r requirements.txt        # install the right torch build for your machine first
python neural_tree_v6/deploy.py models/assignment2/cifar10-augmented-vote-3.pt examples/cifar10_test_0_cat.png
# {'prediction': 'cat', 'votes': [0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 'members': 3}
cd neural_tree_v6 && python -m pytest tests
```

[docs/REPRODUCE.md](docs/REPRODUCE.md) explains how to prepare the datasets and rerun the experiments. Everything ran on one laptop GPU (NVIDIA RTX 2050, 4 GB).

## Scope

The project adapts established ideas (Jayadeva's tree-type network from the course material, CNN features, bagging, double-descent and loss-landscape analysis) and does not claim to invent them. The work lies in the implementation and in controlled experiments that show where the tree helps and where it does not. Test splits were reused across versions, so test results are exploratory; model choices used training and validation data only.
