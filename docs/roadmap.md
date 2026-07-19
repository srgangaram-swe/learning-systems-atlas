# Roadmap

The roadmap is maintained as assigned GitHub issues grouped into six milestones.
Each milestone is a coherent capability increment, not a checklist of unrelated
algorithms.

## Sprint 1 — Foundations and three-paradigm vertical slice

Typed package/config/CLI boundaries, estimator validation, transactional
artifacts, CI, supervised regression/classification, unsupervised clustering,
tabular RL, diagnostic evidence, and end-to-end tests.

[Milestone](https://github.com/srgangaram-swe/comprehensive_ml/milestone/1)

## Sprint 2 — Trees, ensembles, and margins

From-scratch OLS/gradient descent, Ridge/Lasso, logistic regression, CART,
random forests, gradient boosting, k-NN, kernel SVMs, Naive Bayes, shared
metrics/data utilities, and a controlled comparison harness.

[Milestone](https://github.com/srgangaram-swe/comprehensive_ml/milestone/2)

## Sprint 3 — Unsupervised learning

k-means++, GMM/EM, DBSCAN, hierarchical clustering, PCA/SVD, t-SNE, and
evaluation that covers internal quality, external agreement, stability, and
failure cases.

[Milestone](https://github.com/srgangaram-swe/comprehensive_ml/milestone/3)

## Sprint 4 — Deep learning

A minimal reverse-mode autograd engine to demonstrate fundamentals, followed by
a production-shaped PyTorch training loop and tested MLP, CNN, LSTM, and
autoencoder studies.

[Milestone](https://github.com/srgangaram-swe/comprehensive_ml/milestone/4)

## Sprint 5 — Reinforcement learning

Bandits, value/policy iteration, tabular Q-learning/SARSA, REINFORCE, and DQN
with multi-seed evaluation, confidence intervals, and sample-efficiency analysis.

[Milestone](https://github.com/srgangaram-swe/comprehensive_ml/milestone/5)

## Sprint 6 — Benchmarks, documentation, and v1.0

Reference comparisons, guided notebooks as thin package clients, model cards,
API documentation, portfolio polish, release automation, and promotion through
`dev → main → prod`.

[Milestone](https://github.com/srgangaram-swe/comprehensive_ml/milestone/6)

## Definition of done

An implementation work item is complete only when it has typed APIs, explicit
seeds, unit tests, an end-to-end test, a baseline comparison, documented leakage
controls, reproducible artifacts/plots, limitations, and all CI gates passing.
