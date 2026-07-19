# ADR-0002: Publish local artifacts transactionally

- Status: accepted
- Date: 2026-07-12

## Context

Interrupted runs, invalid result declarations, and accidental overwrites make
scientific evidence ambiguous. A hosted tracking service would add credentials
and external availability to the first milestone.

## Decision

Write into a sibling staging directory, validate the complete result/artifact
contract, hash every file, then atomically rename into a previously empty target.
Apply the same pattern to multi-experiment suites.

## Consequences

Local runs are self-contained, portable, testable, and leave no partial
published state. Hosted experiment tracking remains a later adapter rather than
a dependency of the core contract.
