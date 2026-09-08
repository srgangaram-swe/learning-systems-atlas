"""Tabular Q-learning with evaluation isolated from training exploration."""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import seaborn as sns
from gymnasium.spaces import Discrete
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from learning_atlas.core.config import QLearningConfig
from learning_atlas.core.contracts import (
    CandidateResult,
    LearningParadigm,
    RunContext,
    RunResult,
    SourceKind,
    SourceMetadata,
)
from learning_atlas.core.reproducibility import derive_seed, generator_for_seed
from learning_atlas.reporting.plots import publish_figure

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class QLearningAgent:
    """Minimal tabular agent with an explicit Bellman update."""

    q_values: FloatArray
    learning_rate: float
    discount_factor: float

    @classmethod
    def initialize(
        cls,
        n_states: int,
        n_actions: int,
        *,
        learning_rate: float,
        discount_factor: float,
    ) -> "QLearningAgent":
        """Create a zero-initialized finite-state agent."""

        if n_states <= 0 or n_actions <= 0:
            msg = "state and action counts must be positive"
            raise ValueError(msg)
        if not 0.0 < learning_rate <= 1.0:
            msg = "learning_rate must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 <= discount_factor <= 1.0:
            msg = "discount_factor must be in [0, 1]"
            raise ValueError(msg)
        return cls(
            q_values=np.zeros((n_states, n_actions), dtype=np.float64),
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def select_action(
        self,
        state: int,
        epsilon: float,
        generator: np.random.Generator,
    ) -> int:
        """Choose an epsilon-greedy action with unbiased random tie-breaking."""

        if not 0.0 <= epsilon <= 1.0:
            msg = "epsilon must be in [0, 1]"
            raise ValueError(msg)
        if generator.random() < epsilon:
            return int(generator.integers(self.q_values.shape[1]))
        row = self.q_values[state]
        best_actions = np.flatnonzero(np.isclose(row, np.max(row)))
        return int(generator.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        *,
        terminated: bool,
    ) -> None:
        """Apply one off-policy Bellman update, masking true terminal states only."""

        bootstrap = (
            0.0 if terminated else self.discount_factor * float(np.max(self.q_values[next_state]))
        )
        temporal_difference = reward + bootstrap - self.q_values[state, action]
        self.q_values[state, action] += self.learning_rate * temporal_difference


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Return distribution summary from a fixed evaluation seed stream."""

    mean_return: float
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float


def wilson_interval(successes: int, trials: int, *, z_score: float = 1.96) -> tuple[float, float]:
    """Calculate a bounded Wilson score interval for a Bernoulli success rate."""

    if trials <= 0 or not 0 <= successes <= trials:
        msg = "successes and trials must define a non-empty binomial sample"
        raise ValueError(msg)
    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (proportion + z_score**2 / (2.0 * trials)) / denominator
    radius = (
        z_score
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_score**2 / (4.0 * trials**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == trials else min(1.0, center + radius)
    return lower, upper


def epsilon_for_episode(config: QLearningConfig, episode: int) -> float:
    """Return the bounded exponential exploration schedule."""

    if episode < 0:
        msg = "episode must be non-negative"
        raise ValueError(msg)
    return max(config.epsilon_end, config.epsilon_start * config.epsilon_decay**episode)


def _space_sizes(environment: gym.Env[object, object]) -> tuple[int, int]:
    if not isinstance(environment.observation_space, Discrete) or not isinstance(
        environment.action_space, Discrete
    ):
        msg = "tabular Q-learning requires discrete observation and action spaces"
        raise TypeError(msg)
    return int(environment.observation_space.n), int(environment.action_space.n)


def train_agent(config: QLearningConfig) -> tuple[QLearningAgent, FloatArray]:
    """Train on one environment and return per-episode rewards."""

    environment = gym.make(
        config.environment,
        map_name=config.map_name,
        is_slippery=config.is_slippery,
        max_episode_steps=config.max_steps_per_episode,
    )
    try:
        n_states, n_actions = _space_sizes(environment)
        agent = QLearningAgent.initialize(
            n_states,
            n_actions,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
        )
        policy_seed = derive_seed(config.seed, stream=1)
        environment_seed = derive_seed(config.seed, stream=2)
        generator = generator_for_seed(policy_seed)
        environment.action_space.seed(environment_seed)
        rewards = np.zeros(config.training_episodes, dtype=np.float64)
        observation, _ = environment.reset(seed=environment_seed)
        for episode in range(config.training_episodes):
            if episode:
                observation, _ = environment.reset()
            epsilon = epsilon_for_episode(config, episode)
            for _ in range(config.max_steps_per_episode):
                state = int(observation)
                action = agent.select_action(state, epsilon, generator)
                next_observation, reward, terminated, truncated, _ = environment.step(action)
                agent.update(
                    state,
                    action,
                    float(reward),
                    int(next_observation),
                    terminated=terminated,
                )
                rewards[episode] += float(reward)
                observation = next_observation
                if terminated or truncated:
                    break
    finally:
        environment.close()
    return agent, rewards


def evaluate_policy(
    config: QLearningConfig,
    policy: Callable[[int, np.random.Generator], int],
    *,
    environment_stream: int = 3,
    policy_stream: int = 4,
) -> PolicyEvaluation:
    """Evaluate a policy greedily on an independent, reproducible environment stream."""

    environment = gym.make(
        config.environment,
        map_name=config.map_name,
        is_slippery=config.is_slippery,
        max_episode_steps=config.max_steps_per_episode,
    )
    environment_seed = derive_seed(config.seed, stream=environment_stream)
    policy_seed = derive_seed(config.seed, stream=policy_stream)
    generator = generator_for_seed(policy_seed)
    returns = np.zeros(config.evaluation_episodes, dtype=np.float64)
    try:
        environment.action_space.seed(environment_seed)
        observation, _ = environment.reset(seed=environment_seed)
        for episode in range(config.evaluation_episodes):
            if episode:
                observation, _ = environment.reset()
            for _ in range(config.max_steps_per_episode):
                action = policy(int(observation), generator)
                observation, reward, terminated, truncated, _ = environment.step(action)
                returns[episode] += float(reward)
                if terminated or truncated:
                    break
    finally:
        environment.close()

    mean_return = float(np.mean(returns))
    success_count = int(np.count_nonzero(returns > 0.0))
    success_rate = success_count / len(returns)
    interval_low, interval_high = wilson_interval(success_count, len(returns))
    return PolicyEvaluation(
        mean_return=mean_return,
        success_rate=success_rate,
        success_ci95_low=interval_low,
        success_ci95_high=interval_high,
    )


class FrozenLakeBenchmark:
    """Compare a learned Q-table with a random policy on fixed evaluation streams."""

    def __init__(self, config: QLearningConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        agent, training_returns = train_agent(self._config)

        learned = evaluate_policy(
            self._config,
            lambda state, _generator: int(np.argmax(agent.q_values[state])),
        )
        n_actions = agent.q_values.shape[1]
        random_policy = evaluate_policy(
            self._config,
            lambda _state, generator: int(generator.integers(n_actions)),
        )
        learned_metrics = {
            "mean_return": learned.mean_return,
            "success_rate": learned.success_rate,
            "success_ci95_low": learned.success_ci95_low,
            "success_ci95_high": learned.success_ci95_high,
            "improvement_vs_random": learned.success_rate - random_policy.success_rate,
            "training_last_100_mean_return": float(np.mean(training_returns[-100:])),
        }
        candidates = (
            CandidateResult(
                name="random_policy",
                metrics={
                    "mean_return": random_policy.mean_return,
                    "success_rate": random_policy.success_rate,
                    "success_ci95_low": random_policy.success_ci95_low,
                    "success_ci95_high": random_policy.success_ci95_high,
                },
            ),
            CandidateResult(
                name="q_learning",
                metrics=learned_metrics,
            ),
        )

        table_path = "models/q_table.npz"
        with context.artifacts.atomic_target(table_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, q_values=agent.q_values)
        policy_path = "models/greedy_policy.json"
        context.artifacts.write_json(
            policy_path,
            {
                "actions": np.argmax(agent.q_values, axis=1).astype(int).tolist(),
                "action_legend": {"0": "left", "1": "down", "2": "right", "3": "up"},
            },
        )
        learning_curve_path = self._learning_curve(training_returns, context)
        value_policy_path = self._value_policy_plot(agent.q_values, context)
        evaluation_path = self._evaluation_plot(learned, random_policy, context)

        environment_identity = (
            f"{self._config.environment}|{self._config.map_name}|"
            f"slippery={self._config.is_slippery}|"
            f"max_steps={self._config.max_steps_per_episode}"
        )
        return RunResult(
            experiment=self._config.experiment,
            paradigm=LearningParadigm.REINFORCEMENT,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.ENVIRONMENT,
                name=self._config.environment,
                version="gymnasium-1.3",
                fingerprint_sha256=hashlib.sha256(environment_identity.encode()).hexdigest(),
                target_used_for_fit=None,
                details={
                    "map_name": self._config.map_name,
                    "is_slippery": self._config.is_slippery,
                    "state_count": agent.q_values.shape[0],
                    "action_count": agent.q_values.shape[1],
                    "training_episode_count": self._config.training_episodes,
                    "max_steps_per_episode": self._config.max_steps_per_episode,
                    "train_policy_seed": derive_seed(context.seed, stream=1),
                    "train_environment_seed": derive_seed(context.seed, stream=2),
                    "evaluation_environment_seed": derive_seed(context.seed, stream=3),
                    "evaluation_policy_seed": derive_seed(context.seed, stream=4),
                },
            ),
            selected_model="q_learning",
            metrics=learned_metrics,
            candidates=candidates,
            artifacts={
                "q_table": table_path,
                "greedy_policy": policy_path,
                "learning_curve": learning_curve_path,
                "value_policy_plot": value_policy_path,
                "evaluation_plot": evaluation_path,
            },
            notes=(
                "Training and evaluation use separate environments and random streams.",
                "Evaluation is exploration-free; uncertainty is reported over evaluation episodes.",
                "Truncation ends an episode but does not receive terminal-state masking.",
            ),
        )

    @staticmethod
    def _learning_curve(training_returns: FloatArray, context: RunContext) -> str:
        window = min(250, len(training_returns))
        smoothed = np.convolve(training_returns, np.ones(window) / window, mode="valid")
        sns.set_theme(style="whitegrid", palette="colorblind")
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        sns.lineplot(
            x=np.arange(window - 1, len(training_returns)), y=smoothed, linewidth=1.5, ax=axis
        )
        axis.set(
            title=f"Q-learning success, {window}-episode moving average\nseed={context.seed}; {len(training_returns)} training episodes",
            xlabel="Training episode",
            ylabel="Mean return",
            ylim=(-0.02, 1.02),
        )
        axis.grid(alpha=0.2)
        figure.set_layout_engine("constrained")
        return publish_figure(figure, context.artifacts, "plots/q_learning_curve.png")

    @staticmethod
    def _value_policy_plot(q_values: FloatArray, context: RunContext) -> str:
        grid_size = int(math.sqrt(q_values.shape[0]))
        if grid_size**2 != q_values.shape[0]:
            msg = "FrozenLake state count must form a square map"
            raise ValueError(msg)
        state_values = np.max(q_values, axis=1).reshape(grid_size, grid_size)
        actions = np.argmax(q_values, axis=1).reshape(grid_size, grid_size)
        arrows = np.array(["←", "↓", "→", "↑"])

        sns.set_theme(style="whitegrid", palette="colorblind")
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), layout="constrained")
        sns.heatmap(
            state_values, cmap="magma", square=True, ax=axes[0], cbar_kws={"label": "max Q(s, a)"}
        )
        axes[0].set_title("Learned state values")
        symbols = np.where(state_values > 0.0, arrows[actions], "·")
        sns.heatmap(
            state_values > 0.0,
            cmap=["#ffffff", "#d9eaf4"],
            vmin=0,
            vmax=1,
            annot=symbols,
            fmt="",
            cbar=False,
            annot_kws={"color": "black", "fontsize": 18},
            square=True,
            ax=axes[1],
        )
        axes[1].set_title("Persisted greedy policy")
        for axis in axes:
            axis.set(xlabel="column", ylabel="row")
        figure.suptitle(
            f"FrozenLake value function and policy; seed={context.seed}\nDot: zero learned value; arrows: greedy action"
        )
        figure.set_layout_engine("constrained")
        return publish_figure(figure, context.artifacts, "plots/q_value_policy.png")

    @staticmethod
    def _evaluation_plot(
        learned: PolicyEvaluation,
        random_policy: PolicyEvaluation,
        context: RunContext,
    ) -> str:
        evaluations = (random_policy, learned)
        rates = np.array([evaluation.success_rate for evaluation in evaluations])
        errors = np.array(
            [
                [
                    evaluation.success_rate - evaluation.success_ci95_low
                    for evaluation in evaluations
                ],
                [
                    evaluation.success_ci95_high - evaluation.success_rate
                    for evaluation in evaluations
                ],
            ]
        )
        sns.set_theme(style="whitegrid", palette="colorblind")
        figure, axis = plt.subplots(figsize=(6.5, 4.6))
        sns.barplot(
            x=["random policy", "Q-learning"],
            y=rates,
            hue=["random policy", "Q-learning"],
            palette="colorblind",
            legend=False,
            errorbar=None,
            ax=axis,
        )
        axis.errorbar([0, 1], rates, yerr=errors, fmt="none", color="black", capsize=6)
        for index, rate in enumerate(rates):
            axis.text(index, float(rate + errors[1, index]) + 0.02, f"{rate:.1%}", ha="center")
        axis.set(
            title=f"Exploration-free evaluation: Wilson 95% intervals\nseed={context.seed}; independent evaluation streams",
            ylabel="Success rate",
            ylim=(0.0, 1.0),
        )
        axis.grid(axis="y", alpha=0.2)
        figure.set_layout_engine("constrained")
        return publish_figure(figure, context.artifacts, "plots/q_policy_evaluation.png")
