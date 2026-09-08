# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com), and
the project uses [Semantic Versioning](https://semver.org).

## [Unreleased]

## [1.0.1] - 2026-09-07

### Fixed

- Correct the positive-penalty Ridge card to O(nd²+d³) for explicit Gram
  construction and dense solve, including wide inputs; distinguish zero-penalty
  least squares and document normal-equation conditioning. Estimators are unchanged.

## [1.0.0] - 2026-09-07

### Added

- Strict MkDocs Material documentation, source-derived API inventory for every
  public top-level class, family model cards and implementation-specific complexity.
- Three offline, output-free notebook tours executed through nbclient in CI.
- Nine fixed scikit-learn oracle pairs over three seeds, retaining all train/test
  losses, naive baselines and timing samples. Twenty-five of 27 exact held-out
  comparisons pass; two CART tied-partition disagreements remain explicitly visible.
- Seaborn release evidence and release-only GitHub Pages artifact deployment.

### Release scope

The first six sprints provide the typed transactional platform and initial
reference panels (Sprint 1), scratch supervised estimators (Sprint 2), scratch
clustering/representation (Sprint 3), autodiff and neural systems (Sprint 4), the
expanded offline RL control laboratory (Sprint 5), and documentation/reference
qualification (Sprint 6). The experiment boundary remains unchanged. Future
algorithms, distributed and generative milestones remain planned.

## [0.4.0] - 2026-07-19

### Added

- NumPy-only tensor reverse-mode autodiff with broadcasting-aware gradients,
  graph reuse, finite-difference checks, and a nonlinear MLP trained with
  deterministic minibatch SGD and momentum.
- A typed PyTorch Trainer with CPU/device handling, validation-based early
  stopping, finite loss/gradient/state guards, best-weight restoration,
  deterministic shuffling, versioned safe-load checkpoints, and exact resume.
- Validation-selected CNN-versus-MLP image classification, packed-LSTM
  variable-length sequence classification, and bottleneck-autoencoder anomaly
  and representation studies with honest baselines and untouched-test evidence.
- An atomic four-study `benchmark-deep` workflow with aggregate reports,
  checkpoint/model artifacts, and 15 optimization, comparison, masking,
  reconstruction, latent-space, and failure-revealing plots.
- Comprehensive deep-learning unit and integration coverage for mathematical
  identities, shapes/dtypes, malformed state, numerical failures, checkpoint
  compatibility, deterministic replay, CLI/registry dispatch, and transaction
  cleanup.
- Generative AI & Foundation Model Systems milestone with 24 assigned work items
  (#98–#121) spanning autoregressive, variational, adversarial, flow/score, and
  diffusion families plus adaptation, evaluation, inference, safety, and
  portable single-/distributed-device training contracts.

### Changed

- Added the explicit deep-learning paradigm and PyTorch CPU dependency to the
  typed configuration, registry, provenance, package, and CI surface.
- Enforced the raw branch-edge coverage floor explicitly in local and CI gates.
- Prepared package and citation metadata for version `0.4.0` and promoted
  Sprint 4's measured CPU evidence and limitations throughout the portfolio.

## [0.3.0] - 2026-07-19

### Added

- NumPy-only k-means++ with isolated restarts, full-covariance Gaussian-mixture
  EM in log space, deterministic DBSCAN, and single/complete/average/Ward
  agglomerative clustering with a complete linkage record.
- Thin-SVD PCA with deterministic component orientation and whitening contracts,
  plus exact transductive t-SNE with perplexity and numerical-convergence checks.
- Label-free clustering and representation selection, named perturbation and
  sensitivity trials, retrospective-only ARI/NMI, heterogeneous generators, and
  from-scratch internal/external metrics.
- Atomic two-study unsupervised benchmark command, aggregate reports, persisted
  model/assignment state, and twenty convergence, comparison, stability,
  geometry, hierarchy, representation, and failure-revealing plots.
- Distributed ML Systems & Scale Engineering milestone with 20 assigned work
  items spanning collectives, DDP, FSDP2/ZeRO, model parallelism, resilient
  checkpointing, orchestration, observability, security, cost, and open research.

### Changed

- Prepared package and citation metadata for version `0.3.0` and expanded the
  recruiter-facing evidence around unsupervised mathematical and systems limits.

## [0.2.0] - 2026-07-19

### Added

- NumPy-only metrics, truth-bearing generators, deterministic split/fold
  utilities, and leakage-safe preprocessing.
- From-scratch OLS/lstsq and gradient descent, Ridge, coordinate-descent Lasso,
  and stable regularized logistic regression.
- Deterministic CART, random forests with OOB diagnostics, and regression/binary
  gradient boosting with stage-loss histories and early stopping.
- Vectorized k-NN, linear/polynomial/RBF kernel SVM with KKT diagnostics, and
  Gaussian/Multinomial Naive Bayes in log space.
- Atomic two-task comparison command, aggregate JSON/CSV/Markdown reports,
  serialized fitted pipelines, and twelve committed Sprint 2 plots.
- Hypothesis properties, import-boundary enforcement, independent numerical
  oracles, and comprehensive unit/integration/error-path coverage.
- Algorithms, Proofs & Performance milestone with 15 assigned proof-driven work
  items spanning classical algorithms through graph search and Fox multiplication.

### Changed

- Prepared package version `0.2.0` and expanded the recruiter-facing
  evidence gallery to 21 plots.
- Renamed the public repository to `learning-systems-atlas` while retaining the
  product name **Learning Systems Atlas**.

## [0.1.0] - 2026-07-19

### Added

- Typed experiment, result, source, estimator, and validation contracts.
- Transactional run/suite publication with atomic writes and artifact hashes.
- Leakage-safe regression and classification reference benchmarks.
- Label-isolated k-means/DBSCAN structure-discovery benchmark.
- Tabular Q-learning with independent seed streams and Wilson evaluation intervals.
- Config-driven Typer CLI and nine generated diagnostic/comparison plots.
- Python 3.11–3.13 CI matrix, strict quality gates, and 90% coverage floor.
- Architecture, reproducibility, roadmap, governance, and result documentation.
