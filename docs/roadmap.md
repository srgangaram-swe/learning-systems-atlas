# Roadmap

The roadmap is maintained as assigned GitHub issues grouped into nine milestones.
Each milestone is a coherent capability increment, not a checklist of unrelated
algorithms.

## Sprint 1 — Foundations and three-paradigm vertical slice

Typed package/config/CLI boundaries, estimator validation, transactional
artifacts, CI, supervised regression/classification, unsupervised clustering,
tabular RL, diagnostic evidence, and end-to-end tests.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/1)

## Sprint 2 — Trees, ensembles, and margins

From-scratch OLS/gradient descent, Ridge/Lasso, logistic regression, CART,
random forests, gradient boosting, k-NN, kernel SVMs, Naive Bayes, shared
metrics/data utilities, and a controlled comparison harness.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/2)

## Sprint 3 — Unsupervised learning

k-means++, GMM/EM, DBSCAN, hierarchical clustering, PCA/SVD, t-SNE, and
evaluation that covers internal quality, external agreement, stability, and
failure cases.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/3)

## Sprint 4 — Deep learning

Completed in version `0.4.0`: a NumPy reverse-mode autograd engine and nonlinear
MLP, followed by a deterministic/checkpointable PyTorch Trainer and
validation-selected CNN, packed LSTM, and bottleneck-autoencoder studies. The
transactional CPU reference publishes 15 plots and preserves untouched-test boundaries,
resume state, numerical failures, masking invariants, and baseline comparisons.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/4)

## Sprint 5 — Reinforcement learning

Bandits, exact Gridworld modeling, value/policy iteration, tabular Q-learning/SARSA,
REINFORCE, DQN, and PPO-Clip/GAE with independent multi-seed evaluation,
checkpoint recovery, failure diagnostics, and sample-efficiency analysis.
See the [laboratory contract](reinforcement-learning.md). Sprint closure requires
merged evidence, green CI, and protected release promotions; implementation alone
does not imply a completed release.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/5)

## Sprint 6 — Benchmarks, documentation, and v1.0

Reference comparisons, guided notebooks as thin package clients, model cards,
API documentation, portfolio polish, release automation, and promotion through
`dev → prod → main`.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/6)

## Sprint 7 — Algorithms, proofs, and performance

A production-shaped classical algorithms library spanning sorting/search,
data structures, dynamic programming and greedy optimization, graph traversal,
shortest paths including Dijkstra and A-star, Kosaraju SCCs, flow, strings,
geometry, randomized/approximation methods, and numerical matrix algorithms
including a process-grid implementation of Fox multiplication.

Each implementation must link to a formal specification, termination and
correctness proof, asymptotic analysis, adversarial/property/differential tests,
deterministic benchmarks, and visual evidence. A proof compendium audits the
distinction between exact, numerical, and probabilistic correctness.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/14)

## Sprint 8 — Distributed ML systems and scale engineering

A production-shaped distributed-learning substrate spanning deterministic data
sharding, collective semantics, DDP, FSDP2/ZeRO, tensor/sequence/context,
pipeline, and expert parallelism; topology-aware strategy planning; transactional
distributed checkpoints; elastic recovery; scheduler integration; observability,
security, cost, and low-bandwidth optimization research.

The portable definition of done uses one-machine CPU multiprocess references and
numerical-equivalence/fault-injection tests. Accelerator, fabric, and multi-node
results are separate capability profiles with recorded hardware provenance;
analytical or simulated results are never described as measured cluster speedups.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/15)

## Sprint 9 — Generative AI & foundation model systems

Twenty-four assigned work items (#98–#121) define a production-shaped generative
AI slice spanning generative modeling foundations, tokenization and attention,
autoregressive/sequence-to-sequence transformers, variational and adversarial
models, normalizing-flow and energy/score perspectives, diffusion models and
their sampling variants, parameter-efficient adaptation, retrieval/tool
boundaries, evaluation, inference efficiency, observability, safety, and an
end-to-end evidence suite.

Every model family must have explicit mathematical and data contracts,
actionable failure handling, deterministic unit tests, public-API/config/CLI
integration tests, and CPU-capable smoke profiles. Training interfaces must
support one-device execution and define sharding/checkpoint boundaries that can
later integrate with Sprint 8; no multi-node, frontier-scale, quality, cost, or
speedup claim is complete without measured hardware evidence and provenance.

[Milestone](https://github.com/srgangaram-swe/learning-systems-atlas/milestone/16)

## Definition of done

An implementation work item is complete only when it has typed APIs, explicit
seeds, unit tests, an end-to-end test, a baseline comparison, documented leakage
controls, reproducible artifacts/plots, limitations, and all CI gates passing.

An algorithms work item additionally requires a machine-linked proof artifact
covering preconditions, postconditions, termination, correctness, and time/space
complexity. Empirical tests support a proof; they never substitute for one.

A generative or distributed work item additionally requires explicit data
licensing/provenance, checkpoint compatibility, numerical-failure behavior,
single-device tests, distributed-equivalence tests where applicable, evaluation
limitations, and security/abuse considerations appropriate to the capability.
