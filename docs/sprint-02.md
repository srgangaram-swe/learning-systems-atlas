# Sprint 2 — From-scratch supervised systems

Sprint 2 makes the mathematical layer inspectable. The new estimators are
implemented with NumPy and the atlas validation/fitted-state contracts; they do
not import scikit-learn, configuration, plotting, CLI, or artifact code. An AST
architecture test enforces that dependency boundary. Scikit-learn remains an
independent differential-test oracle and a production reference in the earlier
Sprint 1 experiments.

## Experimental boundary

Two task-local experiments preserve scientifically meaningful selection:

- `scratch_regression_benchmark` compares the mean baseline, OLS by least
  squares and gradient descent, Ridge, Lasso, CART, random forest, gradient
  boosting, and distance-weighted k-NN.
- `scratch_classification_benchmark` compares the class-prior baseline,
  regularized logistic regression, CART, random forest, gradient boosting,
  distance-weighted k-NN, linear and RBF kernel SVMs, Gaussian Naive Bayes, and
  Multinomial Naive Bayes.

The outer holdout is split before any learned transform. Every candidate sees
the same seeded inner folds, and each fold learns fresh preprocessing using only
its training indices. The winner is chosen by mean validation RMSE for
regression or macro-F1 for classification. After each candidate's CV evidence is
fixed, its final refit is evaluated once on the holdout for a descriptive
comparison; those test values cannot alter selection. The command below
publishes both child runs and aggregate JSON, CSV, and Markdown reports as one
atomic transaction:

```bash
uv run learning-atlas benchmark-supervised \
  --config-dir configs/supervised/sprint-02 \
  --output-dir runs/sprint-02
```

## Mathematical implementation notes

### Metrics and data

Regression metrics are pure functions for MSE, RMSE, MAE, and the finite
constant-target convention for R-squared. Classification includes accuracy,
macro precision/recall/F1, confusion matrices, and binary or multiclass log
loss. Probabilities are validated, normalized, and clipped only at the logarithm
boundary. Hand fixtures, algebraic properties, and independent-library parity
tests cover the implementations.

Synthetic generators own explicit local random streams and return the latent
parameters that generated each sample: regression coefficients/noise,
classification hyperplane/clean labels, or blob centers. `StandardScaler`
learns population moments, maps constant-column scale to one, and round-trips
through its inverse. Seeded split/fold utilities guarantee disjointness,
completeness, and deterministic stratification.

### Linear and generalized-linear models

OLS solves

```text
minimize_w,b  (1 / 2n) ||Xw + b - y||²
```

with `numpy.linalg.lstsq`, never an explicit inverse. The alternative full-batch
gradient solver derives a stable default step from the spectral Lipschitz bound,
records its objective, and raises a contextual convergence error if its stopping
contract is exhausted.

Ridge centers the design so the intercept is never penalized and solves

```text
(X_centeredᵀ X_centered + alpha I) w = X_centeredᵀ y_centered.
```

Lasso uses cyclic coordinate descent and the soft-threshold operator. `alpha=0`
reduces to OLS, while positive regularization yields exact zeros and is tested
against coordinate-wise optimality conditions.

Binary logistic regression minimizes stable regularized negative log likelihood
using `logaddexp`; the L2 term excludes the intercept. Its sigmoid avoids direct
evaluation of an overflowing exponential. Predictions, normalized probabilities,
loss history, convergence, and fitted class order are explicit state.

### Trees and ensembles

CART enumerates sorted midpoint thresholds and chooses the largest impurity
reduction. Classification supports Gini and entropy; regression uses squared
error. Equal gains resolve by feature index and threshold, making ties a tested
part of the API. Depth, split/leaf sample minima, per-node feature subsampling,
and minimum impurity decrease control growth.

Random forests derive independent child seeds, bootstrap rows, and subsample
features at each split. Out-of-bag predictions use only trees for which a row was
not in-bag and report coverage separately from score; class probabilities are
aligned even when a bootstrap omits a class.

Gradient boosting fits shallow regression trees stagewise to residual or binary
log-loss gradients. Shrinkage, training/validation loss histories, and
patience-based early stopping are observable. Training loss monotonicity is a
tested numerical invariant rather than an assumed plotting artifact.

### Instance, margin, and Bayesian methods

k-NN evaluates query-to-training distances through vectorized NumPy broadcasting
and a stable neighbor order. Euclidean, Manhattan, and general Minkowski metrics
support uniform or inverse-distance weights. Exact-zero neighbors receive all
weight before any nonzero-distance observation, avoiding division artifacts.

The binary kernel SVM solves the C-constrained dual with deterministic sequential
minimal optimization. Linear, polynomial, and RBF kernels expose support indices,
support vectors, dual coefficients, objective, iteration count, convergence, and
maximum KKT violation. The implementation correctly treats `C` as the upper
bound on dual multipliers—not as a bound on the magnitude of margin errors.
Equivalently, the soft-margin primal minimizes

```text
(1 / 2) ||w||² + C sum_i xi_i
subject to y_i f(x_i) >= 1 - xi_i and xi_i >= 0.
```

Each slack variable `xi_i` measures a sample's hinge-margin shortfall; `C`
penalizes their total and induces the dual box constraint `0 <= alpha_i <= C`.
The benchmark requires the fitted KKT residual to satisfy its declared tolerance
and fails atomically otherwise.
SVM scores are not transformed into invented probabilities; probability metrics
are reported only for estimators with a modeled probabilistic output.

Gaussian and Multinomial Naive Bayes compute posteriors in log space and normalize
with log-sum-exp. Gaussian variance smoothing prevents a constant feature from
creating an infinite likelihood. Multinomial likelihoods use Laplace smoothing
and strictly reject negative training/inference values. The benchmark learns a
nonnegative shift inside each fold and documents that this makes the model an
applicability study rather than a claim that Gaussian synthetic features are
natural count data.

## Complexity and limitations

| Family | Dominant fit cost | Dominant inference cost | Deliberate limitation |
|---|---:|---:|---|
| OLS/Ridge | `O(nd² + d³)` | `O(qd)` | dense solve; assumes `n >= d` for this expression |
| linear/logistic GD | `O(Tnd)` | `O(qd)` | convergence depends on scaling |
| Lasso coordinate descent | `O(Tnd)` | `O(qd)` | dense cyclic updates |
| CART | `O(mn²h)` | `O(qh)` | exhaustive reference split search |
| Forest/boosting | `O(Bmn²h)` | `O(Bqh)` | CPU-only, sequential reference profile |
| k-NN | `O(nd)` storage | `O(q(nd + n log n))` | brute-force distances and neighbor ordering |
| kernel SVM | `O(n²d + Tn² log n)` | `O(qsd)` | dense Gram matrix and deterministic SMO scans |
| Naive Bayes | `O(nd + Kd)` | `O(qKd)` | conditional-independence assumption |

Here `n` is training rows, `q` query rows, `d` input features, `m` features
considered per tree node, `h` maximum tree depth, `B` trees or boosting stages,
`T` optimizer/SMO scans, `K` classes, and `s` support vectors. CART is stated for
this clarity-first implementation, which recomputes impurity across candidate
thresholds; optimized sorted-scan implementations have tighter constants and
bounds. The committed datasets are deterministic CPU studies designed for
auditable CI, not production-scale throughput claims.

## Verification contract

- Unit tests cover fitted state, shapes/dtypes, non-finite inputs, invalid
  hyperparameters, deterministic ties, singular/constant cases, serialization,
  and numerical stability.
- Hypothesis tests exercise metric identities and scaler round trips using a
  deterministic bounded profile.
- Differential tests compare predictions, objectives, probabilities, tree
  behavior, kernels, and posteriors with hand calculations, NumPy, or
  scikit-learn at justified tolerances.
- Integration tests execute the public CLI, both real comparison pipelines,
  candidate tables, model serialization, plots, manifests, deterministic replay,
  and atomic cleanup when either task fails.
- CI repeats the complete suite on Python 3.11, 3.12, and 3.13; wall time and
  exact cross-platform floating-point strings are never used as pass criteria.

Observed committed-profile metrics and all plots are recorded in
[the results gallery](results.md) after the generated artifacts are inspected.
