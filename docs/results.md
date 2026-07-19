# Reference results

Command:

```bash
uv run learning-atlas run-all --config-dir configs --output-dir runs/reference
```

All committed configs use seed `42`, bundled/synthetic sources, and CPU execution. Metrics
below are evidence from the locked July 2026 environment. They are benchmark
sanity checks and engineering evidence, not state-of-the-art claims.

## Sprint 1 — three-paradigm vertical slice

### Supervised regression

Dataset: scikit-learn diabetes. The candidate is selected by five-fold CV RMSE
on the training split; the test set is opened only for reporting.

| Candidate | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| mean dummy | 78.13 | 73.22 | 64.01 | -0.012 |
| **Ridge** | **55.44** | **53.63** | **42.86** | **0.457** |
| random forest | 58.71 | 54.37 | 44.14 | 0.442 |

![Regression diagnostics](assets/sprint-01/regression_diagnostics.png)

![Regression model comparison](assets/sprint-01/regression_model_comparison.png)

### Supervised classification

Dataset: scikit-learn Wisconsin diagnostic breast cancer. The candidate is
selected by five-fold training CV ROC AUC. This is a software benchmark and not
a clinical model.

| Candidate | CV ROC AUC | Test ROC AUC | Accuracy | Brier score |
|---|---:|---:|---:|---:|
| prior dummy | 0.500 | 0.500 | 0.632 | 0.233 |
| **logistic regression** | **0.996** | **0.995** | **0.982** | **0.022** |
| random forest | 0.989 | 0.993 | 0.947 | 0.033 |

![Classification diagnostics](assets/sprint-01/classification_diagnostics.png)

![Classification model comparison](assets/sprint-01/classification_model_comparison.png)

### Unsupervised structure discovery

Source: noisy two-moons generator. Selection uses silhouette multiplied by
assigned coverage. Labels are retained outside fitting/selection and used only
to calculate retrospective ARI/NMI.

| Candidate | Selection score | Silhouette | Coverage | Retrospective ARI | Retrospective NMI |
|---|---:|---:|---:|---:|---:|
| **k-means** | **0.492** | **0.492** | 1.000 | 0.457 | 0.362 |
| DBSCAN | 0.388 | 0.391 | 0.992 | **0.983** | **0.962** |

This disagreement is evidence, not an error: an internal convexity-biased
metric does not necessarily measure recovery of non-convex latent structure.

![Clustering comparison](assets/sprint-01/clustering_comparison.png)

![Clustering metrics](assets/sprint-01/clustering_metrics.png)

### Reinforcement learning

Environment: slippery `FrozenLake-v1`, 4×4 map, 12,000 training episodes and
1,000 exploration-free evaluation episodes. Wilson intervals summarize binary
success uncertainty.

| Policy | Success rate | Wilson 95% CI | Improvement |
|---|---:|---:|---:|
| random | 1.3% | 0.8–2.2% | — |
| **tabular Q-learning** | **73.5%** | **70.7–76.1%** | **+72.2 percentage points** |

![Q-learning curve](assets/sprint-01/q_learning_curve.png)

![Q-value and policy](assets/sprint-01/q_value_policy.png)

![Policy evaluation](assets/sprint-01/q_policy_evaluation.png)

## Sprint 2 — from-scratch supervised systems

Command:

```bash
uv run learning-atlas benchmark-supervised \
  --config-dir configs/supervised/sprint-02 \
  --output-dir runs/sprint-02
```

The command publishes both task-local runs and aggregate JSON, CSV, and Markdown
reports atomically. Every number below comes from the committed seed-42 profile.
All candidates use identical training folds and fold-fitted preprocessing. Test
metrics do not participate in model selection.

### From-scratch regression

Source: an auditable linear generator with 480 rows, 10 features, seven nonzero
latent coefficients, and Gaussian noise. Selection minimizes four-fold training
CV RMSE.

| Candidate | CV RMSE ± SD | Test RMSE | Test R² | OOB R² |
|---|---:|---:|---:|---:|
| mean baseline | 34.506 ± 3.358 | 35.878 | -0.009 | — |
| linear OLS (`lstsq`) | 11.755 ± 0.368 | **12.635** | **0.875** | — |
| linear OLS (gradient descent) | 11.755 ± 0.368 | **12.635** | **0.875** | — |
| **Ridge (selected)** | **11.753 ± 0.366** | 12.638 | **0.875** | — |
| Lasso | 11.761 ± 0.360 | 12.658 | 0.874 | — |
| CART | 29.247 ± 1.205 | 28.906 | 0.345 | — |
| random forest | 21.629 ± 2.262 | 23.618 | 0.563 | 0.608 |
| gradient boosting | 21.061 ± 1.599 | 22.101 | 0.617 | — |
| distance-weighted k-NN | 21.497 ± 1.659 | 19.608 | 0.699 | — |

Ridge wins strictly on training-fold evidence and reduces test RMSE by `64.8%`
relative to the mean baseline. The direct and gradient OLS solvers agree to the
displayed precision, which is both a numerical cross-check and an expected
property of this well-conditioned linear problem. The nonlinear models are not
expected to beat the correctly specified linear family here; separate adversarial
tests establish their nonlinear capabilities.

<table>
  <tr>
    <td><img src="assets/sprint-02/scratch_regression_cv.png" alt="Regression cross-validation distributions"></td>
    <td><img src="assets/sprint-02/scratch_regression_comparison.png" alt="Regression validation and holdout comparison"></td>
  </tr>
  <tr>
    <td align="center">All inner-fold scores, including variability and baseline</td>
    <td align="center">Training-only selection evidence beside untouched test RMSE</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-02/scratch_regression_diagnostics.png" alt="Ridge held-out diagnostics"></td>
    <td><img src="assets/sprint-02/scratch_regression_optimization.png" alt="Regression optimization histories"></td>
  </tr>
  <tr>
    <td align="center">Prediction, residual structure, and residual distribution</td>
    <td align="center">OLS, Lasso, and boosting objective trajectories</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-02/scratch_regression_regularization.png" alt="Ridge and Lasso regularization paths"></td>
    <td><img src="assets/sprint-02/scratch_regression_ensembles.png" alt="Regression ensemble evidence"></td>
  </tr>
  <tr>
    <td align="center">Coefficient shrinkage and exact Lasso sparsity</td>
    <td align="center">Forest OOB/holdout agreement and boosting stage loss</td>
  </tr>
</table>

### From-scratch classification

Source: a noisy binary hyperplane generator with 520 rows, eight features, five
informative features, separation `0.9`, and label-flip probability `0.06`.
Selection maximizes four-fold training macro-F1.

| Candidate | CV macro-F1 ± SD | Test macro-F1 | Accuracy | Log loss | OOB accuracy |
|---|---:|---:|---:|---:|---:|
| prior baseline | 0.338 ± 0.000 | 0.337 | 0.508 | 0.693 | — |
| logistic regression | **0.931 ± 0.036** | **0.908** | **0.908** | 0.352 | — |
| CART | 0.864 ± 0.027 | 0.877 | 0.877 | 1.832 | — |
| random forest | 0.928 ± 0.036 | **0.908** | **0.908** | **0.332** | 0.923 |
| gradient boosting | 0.900 ± 0.050 | 0.884 | 0.885 | 0.408 | — |
| distance-weighted k-NN | 0.928 ± 0.035 | **0.908** | **0.908** | 1.024 | — |
| linear SVM | **0.931 ± 0.036** | **0.908** | **0.908** | N/A | — |
| RBF SVM | **0.931 ± 0.036** | 0.900 | 0.900 | N/A | — |
| **GaussianNB (selected)** | **0.931 ± 0.036** | **0.908** | **0.908** | 0.503 | — |
| MultinomialNB | **0.931 ± 0.036** | **0.908** | **0.908** | 0.380 | — |

Several candidates produce identical predictions on the validation folds. The
predeclared deterministic name tie-break selects GaussianNB; the tie is retained
rather than broken with holdout evidence. It reaches test macro-F1 `0.908`, a
gain of `0.571` over the prior baseline, with Brier score `0.091`. Both strict SVM
fits converge within the `0.001` KKT tolerance (maximum final residuals `0.000990`
linear and `0.000931` RBF). Their probability metrics remain `N/A`: applying a
sigmoid to an uncalibrated margin would create a misleading probability claim.

<table>
  <tr>
    <td><img src="assets/sprint-02/scratch_classification_cv.png" alt="Classification cross-validation distributions"></td>
    <td><img src="assets/sprint-02/scratch_classification_comparison.png" alt="Classification validation and holdout comparison"></td>
  </tr>
  <tr>
    <td align="center">Fold dispersion and the naive-prior gap remain visible</td>
    <td align="center">Macro-F1 selection evidence and untouched test result</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-02/scratch_classification_diagnostics.png" alt="Classification ROC confusion and reliability diagnostics"></td>
    <td><img src="assets/sprint-02/scratch_classification_boundaries.png" alt="Projected decision surfaces"></td>
  </tr>
  <tr>
    <td align="center">ROC, confusion matrix, and descriptive reliability</td>
    <td align="center">Common two-feature projection with other inputs fixed</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-02/scratch_classification_optimization.png" alt="Classification optimization histories"></td>
    <td><img src="assets/sprint-02/scratch_classification_ensembles.png" alt="Classification ensemble evidence"></td>
  </tr>
  <tr>
    <td align="center">Stable logistic and stagewise boosting log loss</td>
    <td align="center">Forest OOB/holdout accuracy and boosting stage loss</td>
  </tr>
</table>

Read the [Sprint 2 implementation and mathematical notes](sprint-02.md) for
solver formulations, applicability, complexity, and limitations.
