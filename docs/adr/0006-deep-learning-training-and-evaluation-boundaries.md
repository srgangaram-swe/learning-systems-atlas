# ADR 0006 — Deep-learning training and evaluation boundaries

- Status: accepted
- Date: 2026-07-19

## Context

Sprint 4 needs to demonstrate two kinds of competence without conflating them:
the mathematics of differentiation and neural-network optimization, and the
operational engineering required to train, resume, compare, and audit models.
A single abstraction spanning a NumPy teaching engine, PyTorch modules,
optimizers, checkpoints, datasets, model selection, and artifact publication
would hide ownership and make scientific leakage difficult to detect.

Deep-learning state also has different lifetimes and trust levels:

- a dynamic autograd graph exists for one forward/backward computation;
- estimator state must remain coherent across successful and failed refits;
- runtime state includes optimizer moments, data-order RNG, early-stopping
  counters, and current and best weights;
- validation evidence may choose a candidate, while held-out test evidence must
  not influence that choice; and
- a checkpoint is executable-workflow input with a larger attack and
  compatibility surface than a JSON result.

Reference runs must stay CPU-capable, deterministic, network-independent, and
transactional. Future distributed execution should be possible through an
adapter seam without pretending that Sprint 4 measured scale behavior.

## Decision

### Keep the from-scratch and framework-native paths separate

`deep.autograd` owns a minimal NumPy reverse-mode engine and finite-difference
oracle. `deep.mlp` owns an estimator whose complete forward and backward path
uses that engine. Architecture tests prohibit PyTorch, scikit-learn,
configuration, experiment, artifact, and reporting dependencies from crossing
this boundary.

PyTorch is used explicitly for the shared trainer, CNN, LSTM, and autoencoder.
It is not presented as original autodiff work. Conversely, the NumPy engine is
not expanded into a partial clone of a production framework merely to share a
trainer interface.

The only cross-paradigm execution contract remains
`Experiment.run(RunContext) -> RunResult`. Models retain paradigm-native APIs.

### Give runtime state one owner

`Trainer` owns:

- device resolution and deterministic Torch configuration;
- Adam or momentum-SGD construction;
- the explicitly seeded, zero-worker shuffle loader;
- train/evaluation mode transitions and no-gradient evaluation;
- scalar, shape, finiteness, gradient, and post-update state checks;
- optional gradient clipping;
- sample-weighted train and validation loss;
- validation-based early stopping and best-weight restoration; and
- atomic checkpoint save, strict load, and exact compatible resume.

Task objectives own only batch arity, loss, and task metrics. Models own only
architecture, forward behavior, and domain input validation. Benchmarks own data
partitions, candidates, scientific comparisons, selection metrics, and
artifacts. This prevents each model from inventing a subtly different training
loop or holdout policy.

The scratch MLP has a smaller independent optimizer to keep every derivative
auditable. It must roll back a failed refit. The PyTorch trainer stops on invalid
state and preserves atomic completed-epoch checkpoints, but does not promise to
roll back the caller's mutable in-memory module after a mid-epoch failure. The
outer workflow ensures such a failure cannot publish a partial experiment.

### Make validation the only selection dependency

Each experiment defines exactly one selection field:

- mean validation accuracy for the scratch planar tasks;
- validation accuracy for vision and sequence candidates; and
- validation anomaly AUROC for autoencoder versus PCA reconstruction.

The selector receives only that field plus a declared deterministic tie-break.
Test metrics cannot break ties or alter a fitted candidate. Configurable profile
values and source-defined fixed design choices are both frozen before execution.
The test partition is evaluated once for reporting after selection is fixed.

Autoencoder training, PCA fitting, synthetic anomaly scoring, and selection are
label-free. Digit class labels are reserved for retrospective latent coloring
and silhouette analysis. Validation and test anomalies use independent named
streams and disjoint source partitions.

Static and integration tests enforce the selection keys, disjoint deterministic
partitions, label-use declaration, and acceptance evidence. This boundary is
dependency isolation; it is not a substitute for nested resampling or repeated-
seed uncertainty when making population claims.

### Treat checkpoints as versioned, restricted inputs

Checkpoint schema `learning-atlas-trainer/2` stores current and best model
state, optimizer state, Torch and loader RNG state, history, early-stopping
state, trainer signature, and model shape/dtype signature. Saves use a temporary
file and atomic replacement. The stable in-memory archive avoids encoding a
random temporary filename in PyTorch's ZIP container.

Loads are CPU-mapped and use `weights_only=True`. The loader requires an exact
schema and field set, validates history and early-stopping invariants, rejects
non-finite model and recursively nested optimizer tensor state, validates RNG
states against the runtime, and checks trainer/model/optimizer compatibility
before changing the caller-owned model. Any later restore failure rolls model
and RNG state back. The maximum epoch budget may increase; resume-critical
optimization and runtime settings may not change. A checkpoint at its patience
boundary is terminal and is rejected rather than silently extending an
already-stopped run.

Only trusted local checkpoints are supported. Restricted loading, signatures,
and finite-value checks reduce accidental corruption and arbitrary-object risk;
they do not authenticate provenance or eliminate decompression and resource-
exhaustion threats. The suite manifest supplies integrity hashes, not digital
signatures.

### Pin reference evidence to CPU and named randomness

Committed reference profiles request CPU. Root seeds derive stable namespaces
for data, splits, candidates, trainers, model initialization, shuffling, and
anomaly generation. Torch deterministic algorithms are required, and resume
captures both global and shuffle RNG state. Exact replay is scoped to the same
platform, device, and dependency set.

`device: auto` remains available for local exploration, but results from CUDA or
MPS are not asserted to be bit-identical to CPU. This sprint is single-process
and single-device. Distributed data/model parallelism, rank-local randomness,
collectives, sharded checkpoints, elasticity, and scale measurements belong to
the distributed-systems milestone.

### Publish evidence only as a complete suite

Each experiment writes through its artifact store. The four-experiment workflow
requires the exact Sprint 4 set, builds under a sibling staging directory,
validates results, writes aggregate JSON/CSV/Markdown reports and SHA-256 hashes,
then atomically renames the complete tree. It refuses a non-empty destination
and removes staging state on any failure.

Generated runs and checkpoints remain untracked. Source, configurations, tests,
technical documentation, and explicitly curated evidence are the durable
repository contract.

## Consequences

- The from-scratch mathematical claim remains directly inspectable, while the
  framework-native path demonstrates realistic training-system concerns.
- CNN, LSTM, and autoencoder studies share one tested optimizer, evaluation,
  early-stopping, and resume implementation.
- Validation-only selectors make test leakage mechanically reviewable. Test
  performance can disagree with validation choice without silently changing the
  winner.
- Strict checkpoints reject configuration drift instead of attempting an
  ambiguous partial resume. Growing only the epoch ceiling is intentional.
- CPU reference runs are slower than accelerator runs but reproducible in CI and
  do not require specialized hardware.
- Deterministic kernels and zero loader workers trade throughput for auditability.
- A mid-epoch PyTorch failure may mutate the caller's in-memory module; atomic
  checkpoint and suite boundaries prevent publication, and callers that need
  in-memory rollback must restore the last valid checkpoint.
- One fixed seed and split provide reproducibility, not statistical uncertainty
  or external validity. Larger repeated-seed and distributed studies remain
  future work.
- Checkpoint validation narrows but does not remove the security risk of loading
  hostile files; trusted provenance remains an operational requirement.
