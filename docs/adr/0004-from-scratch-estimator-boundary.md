# ADR 0004 — Auditable from-scratch estimator boundary

- Status: accepted
- Date: 2026-07-19

## Context

The atlas needs to demonstrate mathematical implementation skill without
misrepresenting established libraries as original work. It also needs trusted
references for differential testing and production-oriented benchmark context.

## Decision

Sprint 2 numerical estimators are implemented in dedicated NumPy-only modules.
They depend on the atlas validation and fitted-state contracts but not on
scikit-learn, experiment configuration, plotting, CLI, or artifact storage.
An architecture test enforces that import boundary.

Scikit-learn remains an allowed dependency in the earlier reference experiments
and in tests as an independent behavioral oracle. Comparison experiments invoke
the from-scratch APIs through fresh factories, fit preprocessing inside each
training fold, select only with cross-validation, and reserve the outer holdout
for final evidence.

## Consequences

- The mathematical implementation claim is inspectable in source rather than
  inferred from benchmark labels.
- Estimators remain reusable without the experiment system.
- Differential parity can be asserted at outcomes and objectives while allowing
  legitimate solver and tie-breaking differences.
- The project owns numerical stability, convergence, validation, and complexity
  limitations; these cannot be delegated to an opaque third-party estimator.
- Production reference models and from-scratch teaching/research models remain
  deliberately distinct capabilities.
