# Learning Systems Atlas

**Signal. Structure. Strategy.**

[![CI](https://github.com/srgangaram-swe/learning-systems-atlas/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/srgangaram-swe/learning-systems-atlas/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11–3.13-3776AB)
![Coverage](https://img.shields.io/badge/coverage-97.3%25-2E8B57)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Learning Systems Atlas is a reproducible, production-grade portfolio of
implemented supervised, unsupervised, and reinforcement learning systems, with
milestone-backed plans for deep learning and proof-linked classical algorithms.
It connects mathematical foundations and classical estimators to decision
policies, evidence, and production engineering practice without claiming future
milestones as complete.

The project is organized around evidence rather than isolated demos. Every
reference experiment has a typed configuration, explicit seed streams,
leakage controls, a naive baseline, held-out evaluation, versioned artifacts,
diagnostic plots, and unit plus integration tests. Production logic lives in a
typed `src` package; notebooks are reserved for later presentation layers.

> [!NOTE]
> This is a personal engineering and research portfolio built entirely from
> bundled, synthetic, or openly documented sources. It contains no employer or
> proprietary data and does not represent any employer.

## Latest evidence — 19 from-scratch supervised candidate configurations

Sprint 2 adds auditable NumPy implementations beneath the experiment layer. The
locked seed-42 profile uses identical training folds, fold-fitted preprocessing,
and one untouched outer holdout per task.

| Task | Compared candidates | Training-only selection | Untouched-test evidence |
|---|---|---|---|
| Regression | mean, OLS/lstsq, OLS/GD, Ridge, Lasso, CART, forest, boosting, k-NN | Ridge: CV RMSE `11.753 ± 0.366` | RMSE `12.638`, R² `0.875`, `64.8%` lower RMSE than mean |
| Classification | prior, logistic, CART, forest, boosting, k-NN, linear/RBF SVM, Gaussian/Multinomial NB | GaussianNB after a deterministic CV tie: macro-F1 `0.931 ± 0.036` | macro-F1 `0.908`, accuracy `90.8%`, Brier `0.091` |

SVM probability metrics are intentionally absent: an uncalibrated decision
margin is not presented as a probability. The from-scratch source boundary is
machine-tested to reject hidden scikit-learn imports.

<table>
  <tr>
    <td><img src="docs/assets/sprint-02/scratch_regression_cv.png" alt="Regression cross-validation distributions"></td>
    <td><img src="docs/assets/sprint-02/scratch_classification_cv.png" alt="Classification cross-validation distributions"></td>
  </tr>
  <tr>
    <td align="center">Every regression validation fold, not only an aggregate</td>
    <td align="center">Macro-F1 dispersion across ten candidate configurations</td>
  </tr>
  <tr>
    <td><img src="docs/assets/sprint-02/scratch_regression_regularization.png" alt="Regularization path"></td>
    <td><img src="docs/assets/sprint-02/scratch_classification_boundaries.png" alt="Projected decision surfaces"></td>
  </tr>
  <tr>
    <td align="center">Ridge shrinkage and exact Lasso sparsity</td>
    <td align="center">Common projection of six decision surfaces</td>
  </tr>
  <tr>
    <td><img src="docs/assets/sprint-02/scratch_regression_ensembles.png" alt="Regression ensemble evidence"></td>
    <td><img src="docs/assets/sprint-02/scratch_classification_diagnostics.png" alt="Classification diagnostics"></td>
  </tr>
  <tr>
    <td align="center">OOB/holdout agreement and stagewise loss</td>
    <td align="center">ROC, confusion, and descriptive reliability</td>
  </tr>
</table>

See the [complete 21-plot result gallery](docs/results.md) and
[Sprint 2 mathematical implementation notes](docs/sprint-02.md).

## Sprint 1 cross-paradigm evidence

The locked reference profile uses seed `42` and runs without network access.
These are observed outputs from the committed configs, not target values used
to make the tests pass.

| Paradigm | Candidates | Selection boundary | Held-out/reference evidence |
|---|---|---|---|
| Regression | mean dummy, Ridge, random forest | lowest training-fold CV RMSE | Ridge: R² `0.457`, RMSE `53.63`, `26.8%` lower RMSE than dummy |
| Classification | prior dummy, logistic regression, random forest | highest training-fold CV ROC AUC | Logistic: ROC AUC `0.995`, accuracy `98.2%`, AUC gain `+0.495` over dummy |
| Clustering | k-means, DBSCAN | silhouette × assigned coverage; labels withheld | k-means selection score `0.492`; DBSCAN retrospective ARI `0.983` |
| Reinforcement learning | random policy, tabular Q-learning | exploration-free evaluation stream | Q-learning success `73.5%` (Wilson 95% CI `70.7–76.1%`) vs random `1.3%` |

The clustering result deliberately exposes an important evaluation tension:
silhouette favors convex separation and selects k-means, while withheld labels
show that DBSCAN recovers the non-convex moons far better. The external labels
are never available to fitting or model selection.

<table>
  <tr>
    <td><img src="docs/assets/sprint-01/regression_diagnostics.png" alt="Regression held-out diagnostics"></td>
    <td><img src="docs/assets/sprint-01/classification_diagnostics.png" alt="Classification held-out diagnostics"></td>
  </tr>
  <tr>
    <td align="center">Regression: prediction, residual structure, and error distribution</td>
    <td align="center">Classification: ROC, confusion matrix, and calibration</td>
  </tr>
  <tr>
    <td><img src="docs/assets/sprint-01/clustering_comparison.png" alt="Clustering comparison"></td>
    <td><img src="docs/assets/sprint-01/q_policy_evaluation.png" alt="Q-learning evaluation"></td>
  </tr>
  <tr>
    <td align="center">Unsupervised assignments with truth shown only retrospectively</td>
    <td align="center">Random-policy and Q-learning evaluation with Wilson intervals</td>
  </tr>
</table>

See the complete [reference results and plot gallery](docs/results.md).

## What is implemented

| Area | Current implementation | Planned breadth |
|---|---|---|
| Core math/API | fitted-state contracts; finite/shape/dtype validation; metrics; truth-bearing generators; seeded split/fold utilities; preprocessing | calibration, uncertainty, optimization extensions |
| Supervised | from-scratch OLS/GD, Ridge/Lasso, logistic, CART, forests, boosting, k-NN, kernel SVM, Gaussian/Multinomial NB; production-library references | calibration, uncertainty, large-scale solvers |
| Unsupervised | k-means and DBSCAN comparison with label-isolated evaluation | GMM/EM, hierarchical clustering, PCA/SVD, t-SNE, representation learning |
| Deep learning | architecture and work items defined | reverse-mode autograd, MLP, PyTorch engine, CNN, LSTM, autoencoder |
| Reinforcement learning | tabular Q-learning, random baseline, isolated policy evaluation | bandits, dynamic programming, SARSA, REINFORCE, DQN |
| Algorithms | proof/benchmark platform and 15 comprehensive work items defined | sorting/search, structures, DP/greedy, graphs, strings, geometry, Fox matrix multiplication, randomized/approximation |
| ML systems | locked environments, manifests, hashes, transactional artifacts, CI matrix | tracking, validation, serving, monitoring, release automation |

Deep learning is intentionally listed as planned until its tested milestone is
merged; the repository does not claim implementations that do not yet exist.

## Five-minute start

Prerequisites: Git and Python `3.11`, `3.12`, or `3.13`. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/srgangaram-swe/learning-systems-atlas.git
cd learning-systems-atlas
uv sync --locked --all-groups
uv run learning-atlas list
uv run learning-atlas run-all --config-dir configs --output-dir runs/quickstart
```

Each experiment publishes a self-contained directory with a resolved config,
result, manifest, model/policy state, checksums, and plots. Existing non-empty
output directories are never overwritten.

Useful commands:

```bash
uv run learning-atlas validate configs/supervised/classification.yaml
uv run learning-atlas schema
uv run learning-atlas benchmark-supervised -c configs/supervised/sprint-02 -o runs/sprint-02
uv run learning-atlas run configs/reinforcement/q_learning.yaml -o runs/frozen-lake
make check
```

## Architecture

```mermaid
flowchart LR
    YAML[Strict YAML config] --> ADAPTER[Pydantic discriminated schema]
    ADAPTER --> REGISTRY[Explicit experiment registry]
    REGISTRY --> RUNNER[Transactional runner]
    NUMPY[NumPy-only estimator layer] --> SUP[Supervised experiments]
    RUNNER --> SUP[Supervised experiments]
    RUNNER --> UNSUP[Unsupervised experiments]
    RUNNER --> RL[RL experiments]
    SUP --> STORE[Atomic artifact store]
    UNSUP --> STORE
    RL --> STORE
    STORE --> RESULT[result.json]
    STORE --> MANIFEST[manifest.json + SHA-256]
    STORE --> EVIDENCE[models, policies, and plots]
```

The shared boundary is `Experiment.run(RunContext) -> RunResult`. Paradigms keep
their native interfaces—sklearn pipelines for supervised/clustering models and
an explicit Bellman-update agent for RL—instead of being forced into an
artificial common `fit/predict` hierarchy. From-scratch supervised estimators
have a separate fitted-state and validation contract.

Read [architecture](docs/architecture.md),
[reproducibility](docs/reproducibility.md), and the
[architecture decisions](docs/adr/) for the tradeoffs.

## Scientific and engineering controls

- Preprocessing is fit inside pipelines after train/test separation.
- Regression and classification winners are chosen using training-fold CV;
  held-out scores never select a model.
- Clustering receives features only. Labels are reserved for retrospective ARI/NMI.
- RL uses separately derived policy, transition, training, and evaluation seeds;
  true termination and time-limit truncation have distinct Bellman semantics.
- Results reject NaN/inf, declare only existing in-scope artifacts, and publish
  atomically after full validation.
- Tests assert learning relative to naive baselines with stable margins rather
  than brittle exact floating-point goldens.
- From-scratch numerical modules cannot import scikit-learn; hand fixtures,
  algebraic properties, and independent differential oracles test correctness.
- The committed lock covers macOS/Linux resolution; CI tests Python 3.11–3.13
  on immutable action revisions and an explicit Ubuntu runner.

The local gate currently includes strict Ruff formatting/linting, strict mypy,
`405` unit/integration/property tests, `97.3%` branch-aware coverage, lock
verification, and wheel/sdist builds. CI runs the complete suite across all
supported Python versions.

## Roadmap and branching

The repository has seven assigned milestones and 58 scoped work items:

1. [Foundations and three-paradigm vertical slice](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/1)
2. [Trees, ensembles, and margins](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/2)
3. [Unsupervised learning](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/3)
4. [Deep learning](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/4)
5. [Reinforcement learning](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/5)
6. [Benchmarks, documentation, and v1.0](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/6)
7. [Algorithms, proofs, and performance](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/14)

Feature branches are cut from `dev`, validated through pull requests, and
deleted after merge. Release candidates flow `dev → main → prod`. See
[CONTRIBUTING.md](CONTRIBUTING.md) and the detailed [roadmap](docs/roadmap.md).

## Honest limitations

- Reference profiles use small bundled/synthetic problems for deterministic CPU CI; they
  demonstrate methodology and system design, not state-of-the-art claims.
- The kernel SVM is intentionally a small/medium binary solver with quadratic
  kernel storage; the NumPy CART/ensembles optimize auditability before scale.
- Floating-point values can vary slightly across BLAS, platforms, and library
  releases even with identical seeds and dependency resolution.
- The Wisconsin diagnostic dataset is appropriate for a technical benchmark,
  not a deployable clinical model. No clinical-use claim is made.
- Joblib artifacts must never be loaded from untrusted sources. The manifest
  hashes detect drift but do not make pickle-based formats safe.
- GPU reproducibility, distributed training, model serving, and monitoring are
  future milestones and are not implied by the current repository state.

## License

[MIT](LICENSE) © 2026 Saif Ryan Gangaram
