"""On/off-policy temporal-difference control with explicit episode semantics."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from learning_atlas.core.reproducibility import derive_seed
from learning_atlas.reinforcement.gridworld import Gridworld
from learning_atlas.reinforcement.validation import FloatArray, integer, scalar, vector


def epsilon_greedy(row: FloatArray, epsilon: float, rng: np.random.Generator) -> int:
    """Sample unbiased exact-value ties; do not treat distinct near-values as equal."""

    values = vector(row, "action values")
    scalar(epsilon, "epsilon", 0.0, 1.0)
    if rng.random() < epsilon:
        return int(rng.integers(values.size))
    return int(rng.choice(np.flatnonzero(values == values.max())))


def td_target(reward: float, next_value: float, *, gamma: float, terminated: bool) -> float:
    """A time limit is deliberately absent: truncation must still bootstrap."""

    scalar(reward, "reward", -1e6, 1e6)
    scalar(next_value, "next_value", -1e12, 1e12)
    scalar(gamma, "gamma", 0.0, 1.0)
    if not isinstance(terminated, bool):
        raise ValueError("terminated must be bool")
    return reward if terminated else reward + gamma * next_value


@dataclass(frozen=True, slots=True)
class TabularTrace:
    """Final Q table and unsmoothed per-episode evidence."""

    q_values: FloatArray
    returns: FloatArray
    lengths: FloatArray
    cliff_falls: FloatArray


def train_tabular(
    environment: Gridworld,
    *,
    method: Literal["q_learning", "sarsa"],
    episodes: int,
    seed: int,
    learning_rate: float = 0.5,
    gamma: float = 0.95,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> TabularTrace:
    """O(episodes * max_steps * A) bounded interaction; no evaluation RNG reuse."""

    integer(episodes, "episodes", 1, 100_000)
    integer(seed, "seed", 0, 2**32 - 1)
    scalar(learning_rate, "learning_rate", 1e-6, 1.0)
    scalar(gamma, "gamma", 0.0, 1.0)
    scalar(epsilon_start, "epsilon_start", 0.0, 1.0)
    scalar(epsilon_end, "epsilon_end", 0.0, epsilon_start)
    scalar(epsilon_decay, "epsilon_decay", 1e-6, 1.0)
    if method not in {"q_learning", "sarsa"}:
        raise ValueError("unsupported temporal-difference method")
    if episodes * environment.max_steps > 20_000_000:
        raise ValueError("tabular interaction budget exceeds 20 million transitions")
    rng = np.random.default_rng(derive_seed(seed, stream=1))
    q = np.zeros((len(environment.cells), 4))
    returns, lengths, falls = (np.zeros(episodes) for _ in range(3))
    for episode in range(episodes):
        state, _ = environment.reset(seed=derive_seed(seed, stream=100 + episode))
        epsilon = max(epsilon_end, epsilon_start * epsilon_decay**episode)
        action = epsilon_greedy(q[state], epsilon, rng)
        for step in range(environment.max_steps):
            next_state, reward, terminated, truncated, _ = environment.step(action)
            # At a time limit SARSA samples the physical continuation action,
            # but never carries that action into the freshly reset next episode.
            next_action = 0 if terminated else epsilon_greedy(q[next_state], epsilon, rng)
            continuation = (
                float(q[next_state].max())
                if method == "q_learning"
                else float(q[next_state, next_action])
            )
            target = td_target(reward, continuation, gamma=gamma, terminated=terminated)
            q[state, action] += learning_rate * (target - q[state, action])
            returns[episode] += reward
            falls[episode] += reward == environment.cliff_reward
            lengths[episode] = step + 1
            state, action = next_state, next_action
            if terminated or truncated:
                break
    return TabularTrace(q, returns, lengths, falls)
