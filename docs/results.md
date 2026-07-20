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

## Sprint 3 — from-scratch unsupervised systems

Command:

```bash
uv run learning-atlas benchmark-unsupervised \
  --config-dir configs/unsupervised/sprint-03 \
  --output-dir runs/sprint-03
```

The workflow publishes both studies, their model state, deterministic records,
twenty plots, and aggregate JSON/CSV/Markdown reports as one transaction. All
results below come from the committed seed-42 CPU profile. Generator labels are
not provided to any estimator, hyperparameter decision, stability trial, or
selection score. ARI/NMI and truth-colored panels are retrospective audits
computed after selection.

### Heterogeneous clustering study

The comparison uses 210 observations from each of four deterministic sources:
isotropic blobs, two moons, anisotropic blobs, and variable-density blobs. Each
candidate sees the same features. The label-free score combines mean silhouette,
assigned coverage, and partition agreement under three named feature
perturbations for each of the four datasets. Stability therefore pools 12 trials
per candidate, rather than three trials across the entire suite.

| Candidate | Selection score | Silhouette | Stability ± SD | Coverage | Retrospective ARI | Retrospective NMI |
|---|---:|---:|---:|---:|---:|---:|
| **k-means++ (selected)** | **0.843** | **0.759** | 0.997 ± 0.010 | 1.000 | 0.860 | 0.835 |
| full-covariance GMM | 0.842 | 0.758 | 0.995 ± 0.008 | 1.000 | 0.867 | 0.843 |
| DBSCAN | 0.800 | 0.719 | 0.956 ± 0.085 | 0.983 | 0.856 | 0.877 |
| agglomerative single | 0.822 | 0.727 | 1.000 ± 0.000 | 1.000 | **1.000** | **1.000** |
| agglomerative complete | 0.835 | 0.758 | 0.956 ± 0.099 | 1.000 | 0.864 | 0.841 |
| agglomerative average | 0.828 | 0.743 | 0.972 ± 0.082 | 1.000 | 0.870 | 0.876 |
| agglomerative Ward | 0.814 | 0.743 | 0.890 ± 0.239 | 1.000 | 0.870 | 0.876 |

The internal objective selects k-means++, while withheld truth favors
single-link agglomeration on mean ARI/NMI. That mismatch is expected evidence:
silhouette rewards compact separation and cannot encode every legitimate notion
of cluster structure. DBSCAN's `0.983` coverage also exposes its noise decision
instead of silently scoring discarded observations as assigned clusters.

The machine-readable records retain learned-state diagnostics rather than only
headline scores: every k-means restart objective, GMM weights, means,
covariances and responsibilities, DBSCAN core indices, sample roles,
neighborhood counts and k-distances, and linkage matrices plus cuts for
`k=1…6`. GMM parameter counts and per-dataset AIC/BIC are also recorded; those
information criteria audit each fitted mixture on its own dataset and are not
pooled into the cross-dataset winner.

| GMM dataset | Parameters | AIC | BIC | Mean observed-data log likelihood |
|---|---:|---:|---:|---:|
| isotropic blobs | 17 | 151.606 | 208.507 | -0.280 |
| two moons | 11 | 1066.185 | 1103.003 | -2.486 |
| anisotropic blobs | 17 | 324.208 | 381.108 | -0.691 |
| variable-density blobs | 17 | 218.471 | 275.372 | -0.439 |

The metric audit supplies an independent chance baseline: 96 named random
partition trials have mean ARI `-0.000510` (SD `0.005669`, range
`[-0.010649, 0.019920]`), while a perfect label permutation scores exactly
`1.000`. This checks the adjusted-for-chance semantics without using the study
labels for model selection.

<table>
  <tr>
    <td><img src="assets/sprint-03/clustering_assignments.png" alt="Assignments for seven clustering strategies over four geometries"></td>
    <td><img src="assets/sprint-03/internal_silhouette.png" alt="Internal silhouette comparison"></td>
  </tr>
  <tr>
    <td align="center">Comparable assignments across four incompatible geometries</td>
    <td align="center">Internal compactness evidence available to selection</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/assigned_coverage.png" alt="Assigned-sample coverage comparison"></td>
    <td><img src="assets/sprint-03/retrospective_ari.png" alt="Retrospective adjusted Rand index comparison"></td>
  </tr>
  <tr>
    <td align="center">Noise and unassigned samples remain visible</td>
    <td align="center">Truth is opened only after the winner is fixed</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/clustering_stability.png" alt="Perturbation stability distributions"></td>
    <td><img src="assets/sprint-03/restart_objectives.png" alt="K-means and Gaussian-mixture restart objectives"></td>
  </tr>
  <tr>
    <td align="center">Named perturbation trials reveal partition variability</td>
    <td align="center">Every k-means and EM restart remains auditable, not only the winners</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/kmeans_optimization.png" alt="K-means inertia histories"></td>
    <td><img src="assets/sprint-03/gmm_optimization.png" alt="Gaussian-mixture log-likelihood histories"></td>
  </tr>
  <tr>
    <td align="center">Monotone Lloyd progress with explicit convergence</td>
    <td align="center">Stable log-space EM likelihood trajectories</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/gmm_density.png" alt="Gaussian-mixture density and covariance ellipses"></td>
    <td><img src="assets/sprint-03/dbscan_k_distance.png" alt="DBSCAN sorted neighbor-distance diagnostic"></td>
  </tr>
  <tr>
    <td align="center">Full covariance, component weight, and density geometry</td>
    <td align="center">Sorted sixth-neighbor radii expose where the configured epsilon falls</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/dbscan_sensitivity.png" alt="DBSCAN epsilon sensitivity"></td>
    <td><img src="assets/sprint-03/agglomerative_dendrogram.png" alt="Agglomerative clustering dendrogram"></td>
  </tr>
  <tr>
    <td align="center">Failure-revealing cluster, silhouette, and noise sensitivity to epsilon</td>
    <td align="center">The complete merge history, not only a final cut</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/agglomerative_diagnostics.png" alt="Agglomerative cut sensitivity and work-growth diagnostics"></td>
    <td><img src="assets/sprint-03/random_partition_ari.png" alt="Adjusted Rand index under independent random partitions"></td>
  </tr>
  <tr>
    <td align="center">Cut sensitivity and a deterministic O(n³) work proxy make the scale limit explicit</td>
    <td align="center">Ninety-six named trials verify that chance-adjusted ARI centers near zero</td>
  </tr>
</table>

### Representation-learning study

The second study embeds 160 observations from a six-dimensional, four-component
Gaussian source. The selection score uses only ten-neighbor overlap and pairwise
distance correlation. PCA is an inductive transform; exact t-SNE is explicitly
transductive and quadratic, so the implementation does not invent an
out-of-sample `transform` method.

| Candidate | Selection score | Neighbor preservation | Distance correlation | Retrospective ARI | Retrospective NMI |
|---|---:|---:|---:|---:|---:|
| PCA | 0.615 | 0.521 | **0.989** | 1.000 | 1.000 |
| **t-SNE (selected)** | **0.746** | **0.721** | 0.845 | 1.000 | 1.000 |

PCA retains `0.890` of total variance in two components. The selected t-SNE
fit reaches final KL divergence `0.185`; its binary searches attain the requested
perplexity with maximum absolute error `0.000198`. Pairwise distance correlation
is reported but not over-interpreted: t-SNE optimizes local probability structure,
and apparent global distances or cluster sizes in its map are not metric claims.

<table>
  <tr>
    <td><img src="assets/sprint-03/representation_embeddings.png" alt="PCA and t-SNE embedding comparison"></td>
    <td><img src="assets/sprint-03/representation_metrics.png" alt="Label-free representation metrics"></td>
  </tr>
  <tr>
    <td align="center">Truth-colored views are retrospective presentation only</td>
    <td align="center">Neighborhood and global-distance tradeoffs drive selection</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/pca_diagnostics.png" alt="PCA explained variance and reconstruction diagnostics"></td>
    <td><img src="assets/sprint-03/tsne_optimization.png" alt="t-SNE KL-divergence history"></td>
  </tr>
  <tr>
    <td align="center">Singular spectrum, cumulative variance, and reconstruction</td>
    <td align="center">Early exaggeration and post-exaggeration convergence remain visible</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-03/tsne_affinity_diagnostics.png" alt="t-SNE affinity search and gradient diagnostics"></td>
    <td><img src="assets/sprint-03/tsne_sensitivity.png" alt="t-SNE seed and perplexity sensitivity"></td>
  </tr>
  <tr>
    <td align="center">Per-row perplexity error, affinity precision, and gradient norms audit learned state</td>
    <td align="center">Seed/perplexity sensitivity limits single-map conclusions</td>
  </tr>
</table>

These compact CPU references demonstrate mathematical and systems contracts,
not production-scale clustering or state-of-the-art representation quality.
Pairwise density/manifold operations are quadratic and agglomeration can require
cubic work; those limits motivate the separately planned distributed-systems
milestone rather than an unsupported scale claim.

Read the [Sprint 3 mathematical and engineering notes](sprint-03.md) for the
objectives, update equations, deterministic tie rules, complexity bounds,
failure contracts, seed namespaces, and work-item traceability.

## Sprint 4 — deep-learning systems

Command:

```bash
uv run learning-atlas benchmark-deep \
  --config-dir configs/deep/sprint-04 \
  --output-dir runs/sprint-04
```

The workflow runs four studies and publishes their checkpoints, learned state,
candidate records, 15 plots, aggregate JSON/CSV/Markdown comparisons, and a
SHA-256 manifest as one transaction. Every result below is from the committed
seed-42 CPU profile. Training data fit parameters, validation data select the
candidate and stopping state, and untouched test data are opened only after
selection.

These are compact correctness and engineering references. They use one seed and
one split, the bundled 8×8 digits dataset, synthetic temporal XOR, and synthetic
image corruptions. They do not establish robustness, scaling, or state-of-the-art
quality on natural data.

### Reverse-mode autodiff and from-scratch MLP

The first study implements tensor-valued reverse-mode differentiation without
PyTorch or scikit-learn, then builds linear layers, tanh/ReLU, softmax
cross-entropy, minibatch SGD, and momentum on that engine. On the committed XOR
and two-moons tasks, the MLP reaches mean validation and test accuracy `1.000`;
the linear logistic baseline reaches mean validation accuracy `0.5938` and test
accuracy `0.600`. A frozen
minibatch finite-difference audit bounds the worst analytic-gradient error at
`1.5634e-9`.

| Evidence | Observed value |
|---|---:|
| MLP mean validation accuracy | `1.0000` |
| MLP mean untouched-test accuracy | `1.0000` |
| Logistic mean validation accuracy | `0.5938` |
| Logistic mean untouched-test accuracy | `0.6000` |
| Maximum absolute gradient-check error | `1.5634e-9` |

<table>
  <tr>
    <td><img src="assets/sprint-04/mlp_loss_curves.png" alt="From-scratch MLP optimization histories"></td>
    <td><img src="assets/sprint-04/mlp_decision_boundaries.png" alt="From-scratch MLP decision boundaries"></td>
  </tr>
  <tr>
    <td align="center">Deterministic minibatch loss histories expose convergence</td>
    <td align="center">Nonlinear decisions on XOR and two moons</td>
  </tr>
</table>

The exact scores come from controlled synthetic problems designed to expose
linear and nonlinear behavior. The meaningful evidence is the end-to-end
agreement between a reusable computation graph, finite differences,
optimization history, and a baseline that cannot express the required nonlinear
boundary.

### CNN image classification

The vision study compares a compact CNN with a dense MLP on the same normalized
digits, split, Trainer, optimizer family, and epoch budget. Selection uses
validation accuracy; the test set does not tune the architecture. The CNN wins
on both validation and test evidence.

| Candidate | Validation accuracy | Untouched-test accuracy |
|---|---:|---:|
| vision MLP | `0.9637` | `0.9414` |
| **compact CNN (selected)** | **`0.9793`** | **`0.9707`** |

The observed test gain is `0.0293` (2.93 percentage points). That is a
single-seed result on small 8×8 images, not an uncertainty-qualified claim that
the architecture dominates across datasets or initializations. Checkpoint
reload tests compare model state, logits, loss, and metrics rather than relying
on unchanged accuracy alone.

<table>
  <tr>
    <td><img src="assets/sprint-04/vision_training_loss.png" alt="Vision training and validation loss"></td>
    <td><img src="assets/sprint-04/vision_validation_accuracy.png" alt="Vision validation accuracy"></td>
  </tr>
  <tr>
    <td align="center">Shared-budget loss trajectories for CNN and MLP</td>
    <td align="center">Validation-only selection evidence over epochs</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-04/vision_confusion_matrix.png" alt="Compact CNN held-out confusion matrix"></td>
    <td><img src="assets/sprint-04/vision_misclassified.png" alt="Compact CNN misclassified digits"></td>
  </tr>
  <tr>
    <td align="center">Per-class held-out error structure</td>
    <td align="center">Remaining mistakes are shown rather than hidden by accuracy</td>
  </tr>
  <tr>
    <td colspan="2"><img src="assets/sprint-04/vision_conv_filters.png" alt="Learned first-layer convolution filters"></td>
  </tr>
  <tr>
    <td colspan="2" align="center">Learned first-layer kernels provide a parameter-level diagnostic</td>
  </tr>
</table>

### Variable-length sequence classification

The sequence task makes the target depend on the first and final token of a
variable-length distractor sequence. A packed LSTM receives true lengths; the
baseline MLP consumes a fixed padded layout. The LSTM separates the task while
the baseline remains close to chance.

| Candidate | Validation accuracy | Untouched-test accuracy |
|---|---:|---:|
| padded-position MLP | `0.5052` | `0.5625` |
| **packed LSTM (selected)** | **`1.0000`** | **`1.0000`** |

Appending additional padding changes the LSTM logits by a maximum of `0.0000`.
This is a direct masking/packing invariant, not merely an aggregate accuracy
check. The task is synthetic and deliberately isolates long-range state
retention; it does not represent natural-language performance.

<table>
  <tr>
    <td><img src="assets/sprint-04/sequence_training_loss.png" alt="Sequence training and validation loss"></td>
    <td><img src="assets/sprint-04/sequence_validation_accuracy.png" alt="Sequence validation accuracy"></td>
  </tr>
  <tr>
    <td align="center">Optimization behavior for the padded MLP and packed LSTM</td>
    <td align="center">Validation evidence fixes the winner before test evaluation</td>
  </tr>
  <tr>
    <td colspan="2"><img src="assets/sprint-04/sequence_accuracy_by_length.png" alt="Sequence accuracy by true length"></td>
  </tr>
  <tr>
    <td colspan="2" align="center">Length-stratified accuracy can reveal a hidden long-sequence failure</td>
  </tr>
</table>

### Autoencoder representation and anomaly scoring

The final study trains a bottleneck autoencoder on inlier digit images and uses
reconstruction error to rank injected corruptions. Both candidates reach
validation and untouched-test AUROC `1.000`; the declared exact-tie preference
selects the autoencoder before the test partition is opened.
The autoencoder's mean anomaly-to-inlier reconstruction-error ratio is `9.4343`.
PCA's ratio is `7.4342`. Digit labels do not fit or select either representation;
they enter only the retrospective latent-space audit, whose silhouette is
`0.1947`. AUROC is threshold-free; this study does not estimate or calibrate an
operating threshold.

| Evidence | Observed value |
|---|---:|
| Autoencoder validation AUROC | `1.0000` |
| Autoencoder untouched-test AUROC | `1.0000` |
| PCA validation AUROC | `1.0000` |
| PCA untouched-test AUROC | `1.0000` |
| Autoencoder / PCA error-separation ratio | `9.4343` / `7.4342` |
| Retrospective latent-class silhouette | `0.1947` |

The perfect anomaly ranking reflects easily separated synthetic corruptions.
The modest latent silhouette is retained beside it to prevent the anomaly score
from being misrepresented as uniformly class-separable representation learning.

<table>
  <tr>
    <td><img src="assets/sprint-04/autoencoder_training_loss.png" alt="Autoencoder training and validation loss"></td>
    <td><img src="assets/sprint-04/autoencoder_reconstructions.png" alt="Autoencoder inlier and anomaly reconstructions"></td>
  </tr>
  <tr>
    <td align="center">Reconstruction convergence and validation-loss monitoring across the 60-epoch ceiling</td>
    <td align="center">Inputs and reconstructions expose what the score measures</td>
  </tr>
  <tr>
    <td><img src="assets/sprint-04/autoencoder_error_histogram.png" alt="Autoencoder reconstruction-error distributions"></td>
    <td><img src="assets/sprint-04/autoencoder_roc.png" alt="Autoencoder and PCA anomaly ROC curves"></td>
  </tr>
  <tr>
    <td align="center">Inlier/anomaly overlap remains visible</td>
    <td align="center">Ranking evidence for autoencoder and PCA reconstruction</td>
  </tr>
  <tr>
    <td colspan="2"><img src="assets/sprint-04/autoencoder_latent.png" alt="Autoencoder latent representation"></td>
  </tr>
  <tr>
    <td colspan="2" align="center">Truth-colored latent structure is retrospective and reported with silhouette `0.1947`</td>
  </tr>
</table>

Read the [Sprint 4 mathematical and engineering notes](sprint-04.md) for the
autodiff chain rule, Trainer/checkpoint contracts, leakage boundaries, model
interfaces, error handling, reproducibility scope, and known limitations.
