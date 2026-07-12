# ADR-0001: Keep experiments paradigm-native

- Status: accepted
- Date: 2026-07-12

## Context

Supervised estimators, transductive clustering algorithms, and RL agents have
different state transitions and evaluation semantics. A universal `fit/predict`
base would hide important differences such as DBSCAN's lack of out-of-sample
prediction and RL's environment interaction.

## Decision

Share only `Experiment.run(RunContext) -> RunResult` across paradigms. Retain
sklearn pipelines, clustering `fit_predict`, and explicit RL `select_action` /
`update` APIs inside their domains. Provide a separate estimator contract for
from-scratch supervised algorithms.

## Consequences

Cross-paradigm orchestration and artifacts stay uniform without erasing native
semantics. Some domain logic cannot be generalized, which is intentional.
