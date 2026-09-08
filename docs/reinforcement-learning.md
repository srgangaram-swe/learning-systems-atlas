# Reinforcement-learning laboratory

Sprint 5 connects exploration, exact planning, temporal-difference control, and
neural policy optimization behind the existing experiment interface. It is an
offline, CPU-capable research implementation, not a trading or robotics system.

## Run and verify

```bash
uv sync --locked --all-groups
uv run learning-atlas benchmark-reinforcement \
  configs/reinforcement/sprint-05/reinforcement.yaml \
  --output-dir runs/sprint-05

# Reduced integration profile, not a CartPole solve claim:
uv run learning-atlas benchmark-reinforcement \
  tests/fixtures/reinforcement_ci.yaml --output-dir runs/sprint-05-ci

uv run pytest tests/unit/reinforcement tests/integration/test_sprint05_benchmark.py
make check
```

The reference fixes training seeds 42, 43, 44; budgets of 2,000 REINFORCE,
800 DQN, and 300 PPO episodes; a 500-step time limit; ten validation and one
hundred test episodes per trained policy. These budgets differ: do not compare
episode count alone as sample efficiency. The reports include actual transitions.
CartPole's 475/500 criterion is reported per training seed, not inferred from
one favorable episode. Reduced CI profiles test execution and broad learning;
they are not required to solve the full task.

## Mathematical contracts

### Bandits

Agents see chosen-arm rewards only. Bernoulli Thompson sampling updates
`Beta(1+successes, 1+failures)`. For known-variance Gaussian observations and
a N(0,1) prior, posterior variance is `1/(1+n/sigma²)` and mean is that variance
times `sum(rewards)/sigma²`. Incremental means avoid storing reward histories.
UCB uses `mean + scale sqrt(2 log(t+1)/n)` after pulling each unseen arm; scale
is one for bounded Bernoulli rewards and the declared sigma for Gaussian rewards.
The bounded UCB1 theorem is not applied to unbounded rewards without qualification.

Simulations use common potential reward vectors for paired environment draws,
with separate policy RNGs by algorithm. Cumulative pseudo-regret sums
`max(true_means)-true_mean[chosen_arm]`; it is not realized financial loss.
Finite-horizon regret ordering is not guaranteed. All methods and seeds remain
visible, including variance or unfavorable outcomes. O(TK) time, O(K) agent state.

### Gridworld, planning, and temporal differences

The grid validates a bounded rectangle, one start/goal, walls, optional cliffs,
slip, and finite rewards. Slip mixes the intended action with a uniform random
action. The exact model accumulates probabilities and **conditional** rewards
when multiple physical outcomes collide. A cliff incurs its cost and resets to
start without terminating. ASCII rendering requires no GUI.

Value/policy iteration have explicit residual certificates and a
[convergence argument](proofs/discounted-mdp-planning.md). Q-learning bootstraps
the maximum next action; SARSA bootstraps the sampled next behavior action.
Both suppress bootstrap only for true termination. Time-limit truncation ends
the episode but retains the physical continuation value. The cliff comparison
keeps epsilon=0.1 to expose the difference between greedy and exploratory risk.

### REINFORCE and learned baselines

The actor uses log-softmax score gradients with Monte Carlo reward-to-go and
an independently parameterized learned state-value baseline. Return targets and
advantages are detached. A time limit bootstraps the physical final critic value;
a true termination does not. Advantages are standardized across the whole
rollout, not separately per episode. This is a practical normalized policy-gradient
estimator, not an assertion of unbiased finite-sample normalization. One actor
update consumes each rollout; repeated uncorrected reuse is not REINFORCE.

A separate diagnostic freezes the selected policy and critic, collects fresh
paired trajectories, and compares the trace of sample score-gradient covariance
with and without that learned baseline. Discounted trajectory weights are used
in this assay. No diagnostic trajectory fits the critic. A poorly learned
baseline can increase variance; the measured ratio is reported either way.

### DQN

Uniform replay is a fixed-capacity structure-of-arrays ring, O(1) insertion and
O(batch size) sampling/copying. The target is
`r + gamma (1-terminated) max_a Q_target(next_state,a)`. Targets are detached;
Huber TD loss is optimized through the shared Trainer. A frozen target network
syncs by optimizer-update count; epsilon decays by interaction count.

The factorial ablation changes uniform versus recent-window sampling and frozen
versus online targets. Minibatch sizes, update frequency, and episode ceilings
are held fixed, but survival changes realized data volume. This is an informative
controlled engineering comparison with a stated confound, not a guaranteed
causal stability theorem. Training curves, held-out returns, and losses expose
collapse as well as improvement.

### PPO-Clip and generalized advantages

For frozen old log probabilities, `ratio=exp(logp-old_logp)`. Minimize
`-mean(min(ratio*A, clip(ratio,1-epsilon,1+epsilon)*A))` plus value MSE and minus
an entropy bonus. The minimum is essential for **both** advantage signs.
Minibatch epochs reuse only frozen rollout targets. Excessive approximate KL
stops optimization before the next update; clipping is not a formal trust-region
guarantee. Losses, gradients, model state, and likelihood ratios are checked.

GAE uses two masks:

```text
delta[t] = reward[t] + gamma * (1-terminated[t]) * V(next[t]) - V(state[t])
A[t] = delta[t] + gamma * lambda * (1-boundary[t]) * A[t+1]
```

`boundary` includes termination, truncation, and a rollout cut. A terminal is
always a boundary. This prevents advantage carry across resets while retaining
time-limit bootstrap. GAE is O(T) time/memory. Network update cost scales with
transitions, minibatch epochs, and network parameter count; all are bounded.

## Selection, uncertainty, and artifacts

Validation has a separate named stream and never updates a policy. Ties keep
the earliest best checkpoint. Every candidate is fully trained before test
episodes are opened. The headline method is selected by validation return;
DQN ablations are not extra candidates in that search. Tests deliberately replace
test returns and require unchanged selection.

`evidence.json` contains safe synthetic tables and resolved settings;
`comparison.csv`/`.md` summarize every candidate; twenty PNGs show regret,
posteriors, Bellman residuals, grid values/policies, navigation/cliff learning,
cliff risk, DP gaps, neural learning/validation, test seed means/distributions,
REINFORCE variance, DQN ablation/loss, PPO optimization, sample efficiency, and
interaction cost. Seaborn renders every data mark. Bands show seed SD, not
confidence; interaction-indexed plots draw individual seed trajectories rather
than joining interleaved observations from different policies.

Bootstrap intervals resample **training-seed means**. Three seeds cannot establish
a robust population-level performance guarantee. Episode-level ECDFs are labeled
as clustered observations. An undefined diagnostic has a validity flag, never NaN.
Every output is transactionally published and hashed by the existing runner.

## Recovery, security, and limitations

`save_dqn(session, ArtifactStore(...))` and `load_dqn(path, expected_config)` use
typed JSON, a 32 MiB read/write cap, checksum, finite/shape/config validation,
and newly owned restore state. No pickle or executable model loading is used.
Replay, online/target networks, Adam moments, RNGs, episode/update counters, and
validation selection are preserved at completed episode boundaries. Arbitrary
mid-episode recovery is not supported. Same-platform determinism does not promise
bit equality across Torch versions, CPU architectures, or accelerator kernels.

Inputs are bounded before allocation/optimization. Checkpoints, generated runs,
environments, and caches stay ignored. This sprint adds no credentials, network
data source, action execution service, live trading, distributed training,
continuous actions, or general-purpose environment plugin loader.

## Primary references

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [GAE paper](https://arxiv.org/abs/1506.02438)
- [DQN paper](https://arxiv.org/abs/1312.5602)
- [Gymnasium time-limit semantics](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- [Official CartPole contract](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Seaborn line and sampling-unit semantics](https://seaborn.pydata.org/generated/seaborn.lineplot.html)
- [ADR-0007](adr/0007-reinforcement-interaction-and-evidence-boundaries.md)
