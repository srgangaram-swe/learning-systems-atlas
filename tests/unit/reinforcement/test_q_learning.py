"""Bellman update, exploration, and policy-evaluation tests."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import QLearningConfig
from learning_atlas.core.contracts import LearningParadigm, RunContext
from learning_atlas.reinforcement.q_learning import (
    FrozenLakeBenchmark,
    QLearningAgent,
    epsilon_for_episode,
    evaluate_policy,
    train_agent,
    wilson_interval,
)

pytestmark = pytest.mark.unit


def deterministic_config() -> QLearningConfig:
    return QLearningConfig(
        is_slippery=False,
        training_episodes=1_500,
        evaluation_episodes=100,
        epsilon_decay=0.995,
    )


def test_bellman_update_masks_termination_but_bootstraps_truncation() -> None:
    agent = QLearningAgent.initialize(2, 2, learning_rate=0.5, discount_factor=0.9)
    agent.q_values[1] = [2.0, 4.0]
    agent.update(0, 0, reward=1.0, next_state=1, terminated=False)
    assert agent.q_values[0, 0] == pytest.approx(2.3)

    agent.q_values[0, 1] = 0.0
    agent.update(0, 1, reward=1.0, next_state=1, terminated=True)
    assert agent.q_values[0, 1] == pytest.approx(0.5)


def test_agent_validates_shape_and_epsilon() -> None:
    with pytest.raises(ValueError, match="positive"):
        QLearningAgent.initialize(0, 2, learning_rate=0.1, discount_factor=0.9)
    agent = QLearningAgent.initialize(2, 2, learning_rate=0.1, discount_factor=0.9)
    with pytest.raises(ValueError, match="epsilon"):
        agent.select_action(0, 1.1, np.random.default_rng(1))


@pytest.mark.parametrize(
    ("learning_rate", "discount_factor", "message"),
    [(0.0, 0.9, "learning_rate"), (1.1, 0.9, "learning_rate"), (0.1, 1.1, "discount")],
)
def test_agent_validates_learning_dynamics(
    learning_rate: float,
    discount_factor: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        QLearningAgent.initialize(
            2,
            2,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )


def test_epsilon_schedule_starts_decays_and_reaches_floor() -> None:
    config = deterministic_config()
    assert epsilon_for_episode(config, 0) == config.epsilon_start
    values = [epsilon_for_episode(config, episode) for episode in range(10_000)]
    assert all(left >= right for left, right in pairwise(values))
    assert values[-1] == config.epsilon_end
    with pytest.raises(ValueError, match="episode"):
        epsilon_for_episode(config, -1)


@pytest.mark.parametrize("successes", [0, 50, 100])
def test_wilson_interval_is_bounded_and_non_degenerate(successes: int) -> None:
    low, high = wilson_interval(successes, 100)
    rate = successes / 100
    assert 0.0 <= low <= rate <= high <= 1.0
    assert high > low
    with pytest.raises(ValueError, match="binomial"):
        wilson_interval(1, 0)


def test_seeded_action_sequence_is_reproducible_and_in_bounds() -> None:
    agent = QLearningAgent.initialize(2, 4, learning_rate=0.1, discount_factor=0.9)
    first = [agent.select_action(0, 1.0, np.random.default_rng(seed)) for seed in range(20)]
    second = [agent.select_action(0, 1.0, np.random.default_rng(seed)) for seed in range(20)]
    assert first == second
    assert set(first) <= {0, 1, 2, 3}
    assert len(set(first)) > 1


def test_training_and_evaluation_are_reproducible() -> None:
    config = deterministic_config()
    first_agent, first_returns = train_agent(config)
    second_agent, second_returns = train_agent(config)
    np.testing.assert_array_equal(first_agent.q_values, second_agent.q_values)
    np.testing.assert_array_equal(first_returns, second_returns)

    first_evaluation = evaluate_policy(
        config,
        lambda state, _generator: int(np.argmax(first_agent.q_values[state])),
    )
    second_evaluation = evaluate_policy(
        config,
        lambda state, _generator: int(np.argmax(second_agent.q_values[state])),
    )
    assert first_evaluation == second_evaluation
    assert first_evaluation.success_rate > 0.90


def test_benchmark_outperforms_random_and_serializes_policy(tmp_path: Path) -> None:
    config = deterministic_config()
    result = FrozenLakeBenchmark(config).run(
        RunContext(seed=config.seed, artifacts=ArtifactStore(tmp_path))
    )

    assert result.paradigm is LearningParadigm.REINFORCEMENT
    assert result.metrics["success_rate"] > 0.90
    assert result.metrics["improvement_vs_random"] > 0.80
    q_values = np.load(tmp_path / result.artifacts["q_table"])["q_values"]
    assert q_values.shape == (16, 4)
    policy = json.loads((tmp_path / result.artifacts["greedy_policy"]).read_text())
    assert policy["actions"] == np.argmax(q_values, axis=1).astype(int).tolist()
    replay = evaluate_policy(config, lambda state, _generator: int(policy["actions"][state]))
    assert replay.success_rate == result.metrics["success_rate"]
    assert replay.success_ci95_low == result.metrics["success_ci95_low"]
    assert replay.success_ci95_high == result.metrics["success_ci95_high"]
    assert (tmp_path / result.artifacts["learning_curve"]).stat().st_size > 1_000
    assert (tmp_path / result.artifacts["value_policy_plot"]).stat().st_size > 1_000
    assert (tmp_path / result.artifacts["evaluation_plot"]).stat().st_size > 1_000
