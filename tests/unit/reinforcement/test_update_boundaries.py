"""Independent optimizer-boundary and architecture assertions for neural RL."""

import ast
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from learning_atlas.reinforcement.control import create_model
from learning_atlas.reinforcement.dqn import DQNSession
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.objectives import DQNObjective
from learning_atlas.reinforcement.policy_gradient import Rollout, collect_episode

pytestmark = pytest.mark.unit


def test_dqn_targets_mask_only_termination_and_never_receive_gradients(monkeypatch):
    config = ControlConfig(method="dqn", batch_size=2, learning_starts=2, replay_capacity=2)
    session = DQNSession(config)
    with torch.no_grad():
        for parameter in session.target.parameters():
            parameter.zero_()
        session.target[-1].bias.fill_(2.0)
    for terminated in (True, False):
        session.replay.append(np.zeros(4), 0, 0.5, np.ones(4), terminated)
    captured = []

    def capture(model, optimizer, batch, objective, **kwargs):
        captured.append(batch[-1])
        return 0.0

    monkeypatch.setattr(session.trainer, "train_batch", capture)
    session._update()
    targets = captured[0]
    np.testing.assert_allclose(sorted(targets.tolist()), [0.5, 0.5 + 2 * 0.99])
    assert not targets.requires_grad
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in session.target.parameters()
    )


def test_target_synchronization_uses_optimizer_updates_not_environment_steps(monkeypatch):
    config = ControlConfig(
        method="dqn", batch_size=2, learning_starts=2, replay_capacity=2, target_interval=2
    )
    session = DQNSession(config)
    for _ in range(2):
        session.replay.append(np.zeros(4), 0, 1.0, np.ones(4), False)
    initial = {key: value.clone() for key, value in session.target.state_dict().items()}

    def update(model, *args, **kwargs):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.01)
        return 0.0

    monkeypatch.setattr(session.trainer, "train_batch", update)
    session._update()
    for key, value in session.target.state_dict().items():
        assert torch.equal(value, initial[key])
    session._update()
    for key, value in session.target.state_dict().items():
        assert torch.equal(value, session.model.state_dict()[key])


def test_rollout_keeps_physical_time_limit_observation_value():
    config = ControlConfig(method="ppo", episodes=1, max_steps=1)
    model = create_model(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.critic[-1].bias.fill_(4.0)
    environment = gym.make("CartPole-v1", max_episode_steps=1)
    try:
        rollout = Rollout()
        total = collect_episode(
            environment, model, config, episode=0, rng=np.random.default_rng(7), rollout=rollout
        )
        assert total == 1
        assert rollout.terminated == [False]
        assert rollout.boundaries == [True]
        assert rollout.next_values == [4.0]
        assert float(rollout.batch(config)[3][0]) == pytest.approx(0.01 + 0.99 * 4)
    finally:
        environment.close()


def test_invalid_network_return_fails_before_td_arithmetic():
    class Invalid(nn.Module):
        def forward(self, observations):
            return {"not": "a tensor"}

    with pytest.raises(ValueError, match="Q network"):
        DQNObjective().loss(
            Invalid(), (torch.zeros(2, 4), torch.zeros(2, dtype=torch.int64), torch.zeros(2))
        )


def test_rl_math_does_not_delegate_to_an_rl_library_or_import_presentation():
    root = Path("src/learning_atlas/reinforcement")
    for name in ("bandits.py", "planning.py", "tabular.py", "objectives.py"):
        tree = ast.parse((root / name).read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            module.split(".")[0] in {"stable_baselines3", "ray", "sklearn", "seaborn", "matplotlib"}
            for module in imported
        )
        assert not any(
            module.startswith(("learning_atlas.cli", "learning_atlas.reporting"))
            for module in imported
        )
