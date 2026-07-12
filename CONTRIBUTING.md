# Contributing

Learning Systems Atlas is a personal research-engineering project run with a
production codebase's review and evidence standards.

## Branching model

```text
prod  ←  main  ←  dev  ←  feat/<topic>
```

- `feat/*` branches are cut from `dev`, scoped to one work item or sprint, and
  merged into `dev` through a pull request. They are deleted after merge.
- `dev` is the integration branch. Required CI must be green before merge.
- `main` receives validated release candidates from `dev`.
- `prod` receives tagged releases from `main`.

## Environment

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync --locked --all-groups
uv run pre-commit install
```

`uv.lock` is committed and must not be hand-edited. Add/remove dependencies
through `uv`, then verify `uv lock --check`.

## Change contract

Every model or algorithm change includes:

- Typed public APIs and no machine-specific paths.
- Explicit random streams and deterministic CPU quick profiles.
- A naive/established baseline and leakage-safe evaluation boundary.
- Unit tests for math, shape, state, and error paths.
- An integration test that exercises the real pipeline and proves learning by a
  stable margin rather than an exact cross-platform float.
- Generated metrics/plots from committed configs, plus assumptions and limits.
- No secrets, employer/proprietary data, large datasets, caches, or checkpoints.

Production logic belongs under `src/learning_atlas`. A future notebook must be
a thin presentation client whose underlying experiment is independently tested.

## Quality gate

```bash
make check
```

This verifies the lock, formatting, linting, strict types, unit/integration
tests with branch coverage, and package builds. Subset commands are available as
`make unit`, `make integration`, and `make quality`.

## Commits and pull requests

Use concise [Conventional Commits](https://www.conventionalcommits.org), such as
`feat(supervised): add calibrated logistic regression`. Pull requests must link
the work item, explain scientific/engineering impact, include validation
commands and observed evidence, and state risks or limitations.

Work items live in [issues](https://github.com/srgangaram-swe/comprehensive_ml/issues)
grouped by [milestones](https://github.com/srgangaram-swe/comprehensive_ml/milestones).
