# Reproducibility contract

Reproducibility is treated as recorded evidence, not a universal promise of
bitwise equality across arbitrary hardware and releases.

## Random streams

Every committed experiment starts from a validated unsigned 32-bit root seed.
Libraries receive integer seeds explicitly. RL derives independent child seeds
with NumPy `SeedSequence` namespaces:

| Stream | ID | Purpose |
|---|---:|---|
| training policy | 1 | epsilon-greedy decisions |
| training environment | 2 | stochastic transitions |
| evaluation environment | 3 | common held-out transition stream |
| evaluation policy | 4 | stochastic baseline actions |

No experiment depends on process-global Python or NumPy random state. The CI
process sets `PYTHONHASHSEED=0` before startup for stable hash iteration.

For Gymnasium environments, the first reset receives the derived environment
seed, the action space is seeded explicitly, and later episodes advance the
environment stream rather than resetting it to the same episode.

## Data and split boundaries

- Bundled datasets and synthetic generators keep CI network-independent.
- Source fingerprints include array shape, dtype, and contiguous values.
- Train/test splitting precedes preprocessing.
- Scalers live inside model pipelines and are fit inside CV folds.
- Classification splits are stratified.
- Clustering labels are not passed to fitting and do not determine selection.
- RL training and evaluation use separate environment instances and streams.

## Per-run record

Each run directory contains:

| File | Purpose |
|---|---|
| `resolved_config.json` | exact normalized configuration |
| `result.json` | versioned deterministic metrics and declared artifacts |
| `manifest.json` | UTC time, runtime, platform, Python/direct dependency versions, Git state, config hash, artifact hashes |
| `models/*` | selected pipeline, Q-table, and/or persisted policy |
| `plots/*` | evidence generated from the same run |

The runner checks that the result matches the config/registry, the selected
candidate exists once, selected metrics agree with headline metrics, declared
paths stay inside the run, and every artifact exists. It then hashes every file
except the manifest and atomically publishes the directory.

## Expected equality

On the same platform and locked environment, identical configs are tested to
produce equal `RunResult` objects and identical config/result/model/plot hashes.
Only `created_at` and observed `duration_seconds` are expected to change.

Across platforms, tests use behavioral invariants and documented tolerances:
baseline improvement, valid metric ranges, stable selection rules, and broad
performance floors. Exact floating-point comparisons across different BLAS,
CPU/GPU, compiler, or library versions are explicitly not promised.

## Reproduce Sprint 1

```bash
uv sync --locked --all-groups
uv run learning-atlas run-all --config-dir configs --output-dir runs/reference
make check
```
