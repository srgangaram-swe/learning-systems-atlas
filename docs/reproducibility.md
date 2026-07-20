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

Sprint 2 generators and split utilities accept an integer or an explicit local
`numpy.random.Generator`. Comparison experiments derive SHA-256-namespaced
`SeedSequence` streams for data, splitting, folds, and every candidate/fold or
final refit. Candidate streams are keyed by name, so adding or reordering a
candidate cannot perturb an existing model's data or random state. Random
forests derive independent tree streams, and deterministic CART/SVM tie rules do
not depend on set/hash iteration order.

Sprint 3 uses the same stable named-seed primitive for dataset generation,
candidate fits, estimator restarts, perturbation trials, and t-SNE sensitivity
runs. Streams include the dataset and candidate name rather than a loop index.
Reordering or adding a candidate therefore cannot perturb an existing fit, and
changing the number of restarts cannot rewrite an earlier restart's stream.
Deterministic merge and assignment ties are resolved by documented row/cluster
ordering rather than process hash order.

The committed clustering profile performs three named perturbation trials for
each candidate on each of four datasets. Reported candidate stability therefore
aggregates 12 partition comparisons; the dataset dimension is preserved in
`stability_trials.json` so the pooled mean and standard deviation remain
auditable. The separate ARI chance audit uses 96 independently named random
partition streams and records every seed and score.

Sprint 4 derives separate named streams for data generation/subsampling,
three-way splitting, candidate initialization, Trainer state, and anomaly
corruption. The NumPy autograd MLP owns a local generator. PyTorch reference
profiles seed model initialization explicitly, use a dedicated generator for
DataLoader shuffling, request deterministic algorithms, and pin committed runs
to CPU. A checkpoint captures the current and best model state, optimizer state,
completed history, early-stopping counter, CPU Torch RNG, shuffle-generator RNG,
and model/Trainer signatures. Resume tests compare an uninterrupted run with an
interrupted run exactly on the same platform.
An epoch-budget checkpoint may continue under a larger compatible budget; a
checkpoint that already reached its patience boundary is rejected as terminal.

For Gymnasium environments, the first reset receives the derived environment
seed, the action space is seeded explicitly, and later episodes advance the
environment stream rather than resetting it to the same episode.

## Data and split boundaries

- Bundled datasets and synthetic generators keep CI network-independent.
- Source fingerprints include array shape, dtype, and contiguous values.
- Train/test splitting precedes preprocessing.
- Scalers live inside model pipelines and are fit inside CV folds.
- Sprint 2 candidates share identical seeded fold indices; no estimator may
  choose its own favorable partition.
- Classification splits are stratified.
- Sprint 3 clustering labels are not passed to fitting, stability measurement,
  or selection. They enter only a retrospective ARI/NMI and presentation phase
  after the selected candidate is fixed.
- Sprint 3 representation selection uses neighborhood overlap and distance
  correlation; truth-colored embeddings and downstream ARI/NMI are retrospective.
- Sprint 4 uses disjoint train, validation, and test partitions. Validation loss
  controls early stopping and validation metrics select candidates; test metrics
  are computed only after selection. The autoencoder fits inlier images, while
  injected-corruption indicators are reserved for validation/test ranking. Digit
  class labels enter only the retrospective latent-space evidence.
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

The Sprint 2, Sprint 3, and Sprint 4 multi-study workflows additionally publish
`benchmark_manifest.json`, which hashes the aggregate reports and every child
artifact (including each resolved config and child manifest) before the staging
directory is renamed into place.

Sprint 3 also publishes inspectable learned-state arrays and diagnostic records:
k-means restart objectives; GMM responsibilities, parameters, AIC/BIC, and EM
history; DBSCAN roles, neighborhoods, and k-distances; hierarchical linkages and
multiple cuts; metric chance baselines; and t-SNE affinities, achieved
perplexities, precisions, gradients, and KL history. These files are hashed with
the plots that summarize them.

Sprint 4 publishes from-scratch MLP parameters, versioned PyTorch checkpoints,
candidate records, learning curves, held-out diagnostics, and model-specific
state such as convolution filters and latent representations. Checkpoints are
loaded with PyTorch's restricted weights-only loader, validated field by field,
and rejected when their schema, trainer signature, model/best-state signatures,
history/early-stopping state, runtime-compatible RNG states, or recursively
inspected model/optimizer tensors are incompatible. Optimizer compatibility is
probed before mutating the caller-owned model, and restoration failures roll
model and RNG state back. Serialization uses a stable in-memory archive before
atomic publication so randomized temporary filenames do not alter checkpoint
hashes.

The runner checks that the result matches the config/registry, the selected
candidate exists once, selected metrics agree with headline metrics, declared
paths stay inside the run, and every artifact exists. It then hashes every file
except the manifest and atomically publishes the directory.

## Expected equality

On the same platform and locked environment, identical configs are tested to
produce equal `RunResult` objects and identical config/result/model/plot hashes.
Only `created_at` and observed `duration_seconds` are expected to change.

The Sprint 4 equality claim is deliberately scoped to the locked CPU reference.
Exact equality across CPU, CUDA, MPS, distributed world sizes, or different
PyTorch releases is not promised. Accelerator and distributed profiles require
their own recorded hardware/runtime provenance and numerical tolerances.

Across platforms, tests use behavioral invariants and documented tolerances:
baseline improvement, valid metric ranges, stable selection rules, and broad
performance floors. Exact floating-point comparisons across different BLAS,
CPU/GPU, compiler, or library versions are explicitly not promised.

## Reproduce the reference profiles

```bash
uv sync --locked --all-groups
uv run learning-atlas run-all --config-dir configs --output-dir runs/reference
uv run learning-atlas benchmark-supervised \
  --config-dir configs/supervised/sprint-02 \
  --output-dir runs/sprint-02
uv run learning-atlas benchmark-unsupervised \
  --config-dir configs/unsupervised/sprint-03 \
  --output-dir runs/sprint-03
uv run learning-atlas benchmark-deep \
  --config-dir configs/deep/sprint-04 \
  --output-dir runs/sprint-04
make check
```
