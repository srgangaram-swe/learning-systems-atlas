# ADR 0005 — Label-free unsupervised selection

- Status: accepted
- Date: 2026-07-19

## Context

Synthetic structure-discovery datasets provide labels that are valuable for
retrospective scientific evaluation. Allowing those labels to select cluster
counts, hyperparameters, restarts, embeddings, or a winning method would turn an
unsupervised study into a supervised tuning exercise while retaining a misleading
name. Internal indices alone are also insufficient: they encode geometric
preferences and can reward degenerate coverage or unstable partitions.

## Decision

Sprint 3 fits every estimator on features only. Candidate selection combines a
declared internal silhouette convention, assigned-sample coverage, and partition
agreement under named feature perturbations. Degenerate metrics carry explicit
defined/coverage diagnostics and a finite sentinel because the shared result
schema rejects NaN. Candidate and perturbation seeds are derived by stable names,
so adding or reordering candidates cannot alter existing fits.

Generator labels are passed only to a retrospective evaluation phase after all
fits, internal metrics, stability trials, and the selected candidate are fixed.
ARI, NMI, and retrospective plot coloring cannot influence selection. An
integration test replaces or permutes those labels and requires fitted state,
selection, internal scores, seeds, and label-free artifacts to remain unchanged.

Paradigm-native APIs are retained. K-means, Gaussian mixtures, and PCA expose
inductive prediction/transformation where the mathematics defines one. DBSCAN,
agglomerative clustering, and t-SNE are explicitly transductive and do not invent
out-of-sample behavior.

## Consequences

- External recovery can reveal disagreement with internal selection without
  contaminating the experiment.
- No single method is presented as a universal winner across convex, density,
  anisotropic, hierarchical, and representation-learning structure.
- Stability is evidence about a declared perturbation distribution, not a proof
  of population identifiability.
- Pairwise and hierarchical reference implementations remain deliberately small
  and CPU-capable; their quadratic-or-worse limits are reported rather than hidden.
