# Sprint 1 reference results

Command:

```bash
uv run learning-atlas run-all --config-dir configs --output-dir runs/reference
```

All configs use seed `42`, bundled/synthetic sources, and CPU execution. Metrics
below are evidence from the locked July 2026 environment. They are benchmark
sanity checks and engineering evidence, not state-of-the-art claims.

## Supervised regression

Dataset: scikit-learn diabetes. The candidate is selected by five-fold CV RMSE
on the training split; the test set is opened only for reporting.

| Candidate | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| mean dummy | 78.13 | 73.22 | 64.01 | -0.012 |
| **Ridge** | **55.44** | **53.63** | **42.86** | **0.457** |
| random forest | 58.71 | 54.37 | 44.14 | 0.442 |

![Regression diagnostics](assets/sprint-01/regression_diagnostics.png)

![Regression model comparison](assets/sprint-01/regression_model_comparison.png)

## Supervised classification

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

## Unsupervised structure discovery

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

## Reinforcement learning

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
