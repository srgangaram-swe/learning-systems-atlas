# ADR-0003: Start with network-free reference profiles

- Status: accepted
- Date: 2026-07-12

## Context

CI depending on remote datasets is slower and can fail for reasons unrelated to
model correctness. Large datasets/checkpoints also obscure code review.

## Decision

Sprint 1 uses scikit-learn bundled datasets, a seeded synthetic generator, and
Gymnasium's bundled FrozenLake environment. Full reference profiles remain CPU
bounded and do not download data.

## Consequences

Tests are deterministic and fast, but the results demonstrate methodology—not
large-scale or state-of-the-art performance. Later open datasets must include
version, license, checksum, caching, and tiny offline CI fixtures.
