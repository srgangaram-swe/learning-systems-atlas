# ADR-0007: Keep RL interaction native and share the numerical update boundary

- Status: accepted
- Date: 2026-09-06
- Issues: #30–#35, #129

## Context

An epoch over a fixed dataset cannot represent exploration, replay, terminal
bootstrapping, or on-policy rollouts faithfully. Conversely, separate copies of
the optimizer safety loop would let neural RL drift from the existing deep
learning contract. CartPole is a useful bounded control benchmark, not evidence
of trading profitability, robot safety, or production policy deployment.

## Decision

Keep `Experiment.run(RunContext) -> RunResult` and the existing atomic runner.
Expose `Trainer.create_optimizer` and `Trainer.train_batch`; the existing epoch
loop delegates to the same finite-loss, finite-gradient, clipping, optimizer,
and post-update-state checks. A narrow `BatchObjective` supports RL without
inventing fake supervised datasets. An optimizer failure aborts the owned run;
the single-update method does not promise parameter rollback.

RL owns the interaction schedule and its explicitly named RNG streams. NumPy
implements bandits, finite discounted planning, and tabular control. PyTorch
provides differentiation/tensors, not an RL algorithm implementation. REINFORCE,
DQN, PPO clipping, and GAE remain readable, independently tested mathematics.
The reference is single-process CPU. Scoped Torch RNG/determinism/thread settings
are restored on success and failure; this is not a concurrent training API.

PPO collects bounded groups of complete episodes. This avoids confusing reset
observations with physical terminal observations but gives variable-size
rollouts. DQN recovery occurs at completed episode boundaries: episode-indexed
environment seeds reconstruct the next reset without serializing Gymnasium's
private simulator internals. Checkpoints are bounded, typed, checksummed JSON
containing model/target/Adam/replay/RNG/selection state. They never execute pickle
or resolve external tensor paths. Checksums detect corruption, not forgery.

All candidates finish training/validation selection before the benchmark opens
its separate test stream. Ablations are diagnostics, not extra candidates in the
headline winner search. Every seed remains in the evidence. Bootstrap summaries
resample policy-seed means, not episodes treated as independent trained models.
Three-seed intervals are descriptive and fragile; no population-level certainty
is inferred from them.

Seaborn is a justified plotting dependency. Its pandas/tzdata transitive additions
are locked; no training library, environment service, or network dataset is added.
Twenty data-derived plots and JSON/CSV/Markdown reports use the existing
transaction and hash manifest. Model checkpoints stay in ignored run directories.

## Alternatives and consequences

- A universal `fit/predict` wrapper obscures RL state; rejected under ADR-0001.
- A third-party RL trainer hides the mathematics this sprint must demonstrate.
- Arbitrary mid-episode recovery would couple checkpoints to simulator internals.
- Vectorized actors and continuous actions are later extensions, not implied
  capabilities of this categorical four-observation reference.
- DQN ablations share episode ceilings and update settings, not equal realized
  transitions: longer survival itself changes data volume. Report this confound
  rather than labeling the comparison a clean causal estimate of stability.

Rollback is a reviewed revert of the sprint slice. Existing FrozenLake configs,
deep experiment interfaces, Trainer checkpoint schema, coverage gates, and CI
protections remain unchanged.
