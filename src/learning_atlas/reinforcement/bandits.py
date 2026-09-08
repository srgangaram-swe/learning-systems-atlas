"""Stationary bandits with conjugate posteriors and paired potential outcomes.

Agents receive observations, never the true means. Simulation costs O(T K),
including independent potential rewards for all arms at each step. Gaussian UCB
uses the declared known sub-Gaussian scale; UCB1's bounded-reward theorem is not
incorrectly applied to unbounded Gaussian rewards.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from learning_atlas.core.reproducibility import derive_seed
from learning_atlas.reinforcement.validation import FloatArray, integer, scalar, vector

Distribution = Literal["bernoulli", "gaussian"]
BanditMethod = Literal["epsilon_greedy", "ucb", "thompson"]


class BanditAgent:
    """Incremental means and Beta/Normal conjugate posterior statistics, O(K) state."""

    def __init__(self, arms: int, distribution: Distribution, *, sigma: float = 1.0) -> None:
        integer(arms, "arms", 2, 128)
        if distribution not in {"bernoulli", "gaussian"}:
            raise ValueError("unsupported bandit distribution")
        self.distribution = distribution
        self.sigma = scalar(sigma, "sigma", 1e-6, 100.0)
        self.counts = np.zeros(arms, dtype=np.int64)
        self.means = np.zeros(arms, dtype=np.float64)
        self.successes = np.zeros(arms, dtype=np.float64)

    def posterior(self) -> tuple[FloatArray, FloatArray]:
        """Return Beta(alpha,beta), or Normal(mean,variance) from a N(0,1) prior."""

        if self.distribution == "bernoulli":
            return 1.0 + self.successes, 1.0 + self.counts - self.successes
        variance = 1.0 / (1.0 + self.counts / self.sigma**2)
        mean = variance * self.means * self.counts / self.sigma**2
        return mean, variance

    def update(self, arm: int, reward: float) -> None:
        """Validate the observation before changing any posterior statistic."""

        integer(arm, "arm", 0, len(self.counts) - 1)
        scalar(reward, "reward", -1e6, 1e6)
        if self.distribution == "bernoulli" and reward not in {0.0, 1.0}:
            raise ValueError("Bernoulli rewards must be zero or one")
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]
        self.successes[arm] += reward

    def choose(
        self, method: BanditMethod, rng: np.random.Generator, *, epsilon: float = 0.1
    ) -> int:
        """Select using only past observations; break optimistic ties reproducibly."""

        scalar(epsilon, "epsilon", 0.0, 1.0)
        if method == "epsilon_greedy":
            if rng.random() < epsilon:
                return int(rng.integers(len(self.counts)))
            scores = self.means
        elif method == "ucb":
            unseen = np.flatnonzero(self.counts == 0)
            if unseen.size:
                return int(unseen[0])
            scale = 1.0 if self.distribution == "bernoulli" else self.sigma
            scores = self.means + scale * np.sqrt(
                2.0 * np.log(float(self.counts.sum()) + 1.0) / self.counts
            )
        elif method == "thompson":
            first, second = self.posterior()
            scores = (
                rng.beta(first, second)
                if self.distribution == "bernoulli"
                else rng.normal(first, np.sqrt(second))
            )
        else:
            raise ValueError("unsupported bandit method")
        return int(rng.choice(np.flatnonzero(scores == scores.max())))


@dataclass(frozen=True, slots=True)
class BanditTrace:
    """Per-interaction evidence; regret is expected opportunity cost, not lost cash."""

    actions: NDArray[np.int64]
    rewards: FloatArray
    cumulative_regret: FloatArray
    posterior_first: FloatArray
    posterior_second: FloatArray


def simulate_bandit(
    means: FloatArray,
    *,
    distribution: Distribution,
    method: BanditMethod,
    steps: int,
    seed: int,
    sigma: float = 1.0,
    epsilon: float = 0.1,
) -> BanditTrace:
    """Use common environment draws across methods, with isolated policy streams."""

    truth = vector(means, "means")
    integer(steps, "steps", 1, 100_000)
    integer(seed, "seed", 0, 2**32 - 1)
    agent = BanditAgent(len(truth), distribution, sigma=sigma)
    if distribution == "bernoulli" and np.any((truth < 0) | (truth > 1)):
        raise ValueError("Bernoulli means must be probabilities")
    if np.max(np.abs(truth)) > 1e4:
        raise ValueError("bandit means exceed the reward budget")
    methods = ("epsilon_greedy", "ucb", "thompson")
    if method not in methods:
        raise ValueError("unsupported bandit method")
    scalar(epsilon, "epsilon", 0.0, 1.0)
    environment = np.random.default_rng(derive_seed(seed, stream=1))
    policy = np.random.default_rng(derive_seed(seed, stream=10 + methods.index(method)))
    actions = np.empty(steps, dtype=np.int64)
    rewards = np.empty(steps)
    regret = np.empty(steps)
    for step in range(steps):
        potential = (
            (environment.random(len(truth)) < truth).astype(float)
            if distribution == "bernoulli"
            else environment.normal(truth, sigma)
        )
        arm = agent.choose(method, policy, epsilon=epsilon)
        reward = float(potential[arm])
        agent.update(arm, reward)
        actions[step], rewards[step], regret[step] = arm, reward, truth.max() - truth[arm]
    first, second = agent.posterior()
    return BanditTrace(actions, rewards, np.cumsum(regret), first, second)
