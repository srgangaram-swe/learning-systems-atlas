"""Policy and value objectives checked against independent hand calculations."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from learning_atlas.reinforcement.control import (
    action_for,
    create_model,
    create_trainer,
    deterministic_cpu,
    evaluate_control,
    observation_tensor,
)
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.objectives import (
    ActorCritic,
    DQNObjective,
    PolicyObjective,
    checked_tensor,
    clipped_surrogate,
    generalized_advantage,
    policy_statistics,
)
from learning_atlas.reinforcement.policy_gradient import Rollout, train_policy_gradient
from learning_atlas.reinforcement.replay import ReplayBuffer

pytestmark = pytest.mark.unit


def test_gae_hand_calculation_distinguishes_termination_and_time_limit():
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.5, 0.6, 0.7])
    next_values = np.array([0.6, 4.0, 100.0])
    terminated = np.array([False, False, True])
    boundaries = np.array([False, True, True])
    advantages, returns = generalized_advantage(
        rewards, values, next_values, terminated, boundaries, gamma=0.9, gae_lambda=0.8
    )
    # delta1=2+.9*4-.6=5; boundary1 blocks delta2; terminal2 drops Vnext=100.
    np.testing.assert_allclose(advantages, [1.04 + 0.72 * 5, 5, 2.3])
    np.testing.assert_allclose(returns, advantages + values)
    with pytest.raises(ValueError, match="boundary"):
        generalized_advantage(
            rewards,
            values,
            next_values,
            terminated,
            np.zeros(3, dtype=bool),
            gamma=0.9,
            gae_lambda=0.8,
        )
    with pytest.raises(ValueError, match="aligned bool"):
        generalized_advantage(
            rewards,
            values,
            next_values,
            terminated.astype(int),
            boundaries,
            gamma=0.9,
            gae_lambda=0.8,
        )


@pytest.mark.parametrize(
    "advantage,ratio,expected,gradient_zero",
    [
        (2.0, 1.5, -2.4, True),
        (2.0, 0.5, -1.0, False),
        (-2.0, 1.5, 3.0, False),
        (-2.0, 0.5, 1.6, True),
    ],
)
def test_ppo_both_advantage_signs_and_detached_targets(advantage, ratio, expected, gradient_zero):
    current = torch.tensor([np.log(ratio)], requires_grad=True)
    old = torch.zeros(1, requires_grad=True)
    adv = torch.tensor([advantage], requires_grad=True)
    loss, metrics = clipped_surrogate(current, old, adv, 0.2)
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert (float(current.grad[0]) == 0) == gradient_zero
    assert old.grad is None and adv.grad is None
    assert metrics["clip_fraction"] == 1
    assert metrics["approximate_kl"] == pytest.approx(ratio - 1 - np.log(ratio))


def test_ppo_objective_torch_gradcheck_and_unsafe_ratio():
    old = torch.tensor([-0.7, -0.5], dtype=torch.double)
    current = torch.tensor([-0.65, -0.55], dtype=torch.double, requires_grad=True)
    advantages = torch.tensor([1.0, -2.0], dtype=torch.double)
    assert torch.autograd.gradcheck(
        lambda x: clipped_surrogate(x, old, advantages, 0.2)[0], (current,)
    )
    with pytest.raises(ValueError, match="safe numeric"):
        clipped_surrogate(torch.tensor([61.0]), torch.zeros(1), torch.ones(1), 0.2)
    with pytest.raises(ValueError, match="rank one"):
        clipped_surrogate(torch.zeros(1, 1), torch.zeros(1), torch.ones(1), 0.2)


def test_log_probability_and_entropy_stable_under_large_common_offset():
    actions = torch.tensor([0, 1])
    logits = torch.tensor([[10000.0, 10000.0], [-10000.0, -10000.0]])
    chosen, entropy = policy_statistics(logits, actions)
    np.testing.assert_allclose(chosen.numpy(), -np.log(2), rtol=1e-6)
    np.testing.assert_allclose(entropy.numpy(), np.log(2), rtol=1e-6)


@pytest.mark.parametrize(
    "logits,actions",
    [
        (torch.zeros(2), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(2, 2), torch.zeros(2)),
        (torch.zeros(2, 2), torch.tensor([0, 2])),
        (torch.full((2, 2), float("nan")), torch.tensor([0, 1])),
    ],
)
def test_policy_rejects_invalid_logits_and_actions(logits, actions):
    with pytest.raises(ValueError):
        policy_statistics(logits, actions)


def test_actor_critic_detachment_and_shared_trainer_update():
    config = ControlConfig(method="ppo", episodes=2)
    model = create_model(config)
    trainer = create_trainer(config)
    optimizer = trainer.create_optimizer(model)
    advantages, returns, old = (torch.ones(4, requires_grad=True) for _ in range(3))
    batch = (torch.zeros(4, 4), torch.tensor([0, 1, 0, 1]), advantages, returns, old)
    objective = PolicyObjective(ppo=True)
    before = next(model.parameters()).detach().clone()
    trainer.train_batch(model, optimizer, batch, objective, step=1)
    assert advantages.grad is None and returns.grad is None and old.grad is None
    # First weight sees zero input; biases and later layers still update.
    assert torch.equal(before, next(model.parameters()))
    assert objective.diagnostics["entropy"] > 0
    with pytest.raises(ValueError, match="step"):
        trainer.train_batch(model, optimizer, batch, objective, step=0)
    assert np.isfinite(float(PolicyObjective(ppo=False).loss(model, batch).detach()))
    with pytest.raises(ValueError, match="five"):
        objective.loss(model, batch[:3])
    with pytest.raises(ValueError, match="matrix"):
        objective.loss(model, (torch.zeros(4), *batch[1:]))
    with pytest.raises(ValueError, match="actor-critic"):
        objective.loss(nn.Linear(4, 2), batch)


def test_dqn_huber_and_detached_targets():
    model = nn.Linear(4, 2)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    targets = torch.tensor([0.5, 2.0], requires_grad=True)
    batch = (torch.zeros(2, 4), torch.tensor([0, 1]), targets)
    loss = DQNObjective().loss(model, batch)
    assert float(loss.detach()) == pytest.approx((0.125 + 1.5) / 2)
    loss.backward()
    assert targets.grad is None
    with pytest.raises(ValueError, match="three"):
        DQNObjective().loss(model, batch[:2])
    with pytest.raises(ValueError, match="matrix"):
        DQNObjective().loss(model, (torch.zeros(2), *batch[1:]))


def test_replay_ring_ownership_uniformity_and_recent_ablation():
    replay = ReplayBuffer(4)
    observation = np.ones(4)
    for value in range(6):
        replay.append(observation * value, value % 2, float(value), observation, value % 2 == 0)
    observation[:] = 999
    assert replay.position == 2 and replay.size == 4
    recent = replay.sample(3, np.random.default_rng(1), uniform=False)
    np.testing.assert_array_equal(recent[2], [3, 4, 5])
    sampled = replay.sample(4, np.random.default_rng(1))
    assert set(sampled[2].tolist()) == {2, 3, 4, 5}
    assert float(sampled[3].max()) == 1.0
    before = replay.position
    with pytest.raises(ValueError, match="bool"):
        replay.append(np.zeros(4), 0, 0.0, np.ones(4), 1)
    assert replay.position == before
    with pytest.raises(ValueError):
        replay.sample(5, np.random.default_rng())


def test_global_rng_and_thread_settings_restored_after_failure():
    before = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="injected"):
        with deterministic_cpu(123):
            torch.rand(2)
            raise RuntimeError("injected")
    assert torch.equal(before, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    with pytest.raises(ValueError):
        observation_tensor(np.array([1e7] * 4))
    model = ActorCritic()
    with pytest.raises(ValueError, match="optional value"):
        action_for(nn.Linear(4, 1), np.zeros(4), np.random.default_rng(), stochastic=False)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        action_for(model, np.zeros(4), np.random.default_rng(), stochastic=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes": 10000},
        {"batch_size": 64, "replay_capacity": 32},
        {"epsilon_start": 0.1, "epsilon_end": 0.2},
        {"gamma": float("nan")},
    ],
)
def test_config_rejects_unsafe_budgets(kwargs):
    with pytest.raises(ValidationError):
        ControlConfig(method="ppo", **kwargs)


def test_empty_rollout_wrong_method_and_repeatable_random_evaluation():
    with pytest.raises(ValueError, match="empty"):
        Rollout().batch(ControlConfig(method="ppo"))
    with pytest.raises(ValueError, match="requires"):
        train_policy_gradient(ControlConfig(method="dqn"))
    first = evaluate_control(None, seed=8, episodes=10)
    np.testing.assert_array_equal(first, evaluate_control(None, seed=8, episodes=10))
    assert 5 < first.mean() < 100
    with pytest.raises(ValueError):
        checked_tensor(torch.zeros(1), "x", rank=2, length=1)
