"""Uniform-replay DQN with target synchronization and episode-boundary recovery."""

import copy

import gymnasium as gym
import numpy as np
import torch

from learning_atlas.core.reproducibility import derive_seed
from learning_atlas.reinforcement.control import (
    ControlReport,
    ValidationSelector,
    action_for,
    create_model,
    create_trainer,
    deterministic_cpu,
)
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.objectives import DQNObjective
from learning_atlas.reinforcement.replay import ReplayBuffer
from learning_atlas.reinforcement.validation import integer


class DQNSession:
    """Owned mutable training state; public continuation occurs between complete episodes.

    The environment is reconstructed and seeded by absolute episode index. Thus
    exact same-platform resume needs no unsafe Gymnasium private-state capture.
    Selection state is separate from online and target networks.
    """

    def __init__(self, config: ControlConfig) -> None:
        if config.method != "dqn":
            raise ValueError("DQNSession requires method=dqn")
        self.config = config
        self.model = create_model(config)
        self.target = copy.deepcopy(self.model).eval()
        self.target.requires_grad_(False)
        self.trainer = create_trainer(config)
        self.optimizer = self.trainer.create_optimizer(self.model)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.action_rng = np.random.default_rng(derive_seed(config.seed, stream=2))
        self.replay_rng = np.random.default_rng(derive_seed(config.seed, stream=3))
        self.selector = ValidationSelector(config, self.model)
        self.returns: list[float] = []
        self.diagnostics: list[dict[str, float]] = []
        self.environment_steps = 0
        self.updates = 0

    @property
    def epsilon(self) -> float:
        """Linear decay indexed by interaction count, preserved across continuation."""

        fraction = min(1.0, self.environment_steps / self.config.epsilon_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _update(self) -> float:
        batch = self.replay.sample(
            self.config.batch_size, self.replay_rng, uniform=self.config.replay_enabled
        )
        observations, actions, rewards, next_observations, terminated = batch
        target_network = self.target if self.config.target_enabled else self.model
        with torch.no_grad():
            continuation = target_network(next_observations).max(dim=1).values
            targets = rewards + self.config.gamma * (~terminated) * continuation
        self.updates += 1
        loss = self.trainer.train_batch(
            self.model,
            self.optimizer,
            (observations, actions, targets),
            DQNObjective(),
            step=self.updates,
        )
        if self.config.target_enabled and self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.model.state_dict())
        return loss

    def advance(self, episodes: int) -> None:
        """Advance to an absolute episode count without changing configured final budget."""

        integer(episodes, "episodes", len(self.returns) + 1, self.config.episodes)
        environment = gym.make("CartPole-v1", max_episode_steps=self.config.max_steps)
        try:
            with deterministic_cpu(self.config.seed):
                for episode in range(len(self.returns), episodes):
                    observation, _ = environment.reset(
                        seed=derive_seed(self.config.seed, stream=1000 + episode)
                    )
                    total, losses = 0.0, []
                    for _ in range(self.config.max_steps):
                        action = (
                            int(self.action_rng.integers(2))
                            if self.action_rng.random() < self.epsilon
                            else action_for(
                                self.model, observation, self.action_rng, stochastic=False
                            )
                        )
                        next_observation, reward, terminated, truncated, _ = environment.step(
                            action
                        )
                        self.replay.append(
                            observation,
                            action,
                            float(reward) * self.config.reward_scale,
                            next_observation,
                            bool(terminated),
                        )
                        self.environment_steps += 1
                        if (
                            self.replay.size
                            >= max(self.config.learning_starts, self.config.batch_size)
                            and self.environment_steps % self.config.train_frequency == 0
                        ):
                            losses.append(self._update())
                        total += float(reward)
                        observation = next_observation
                        if terminated or truncated:
                            break
                    self.returns.append(total)
                    self.diagnostics.append(
                        {
                            "episode": float(episode + 1),
                            "environment_steps": float(self.environment_steps),
                            "optimizer_updates": float(self.updates),
                            "epsilon": self.epsilon,
                            "td_loss": float(np.mean(losses)) if losses else 0.0,
                            "loss_observed": float(bool(losses)),
                        }
                    )
                    if (
                        episode + 1
                    ) % self.config.validation_interval == 0 or episode + 1 == self.config.episodes:
                        self.selector.consider(self.model, episode + 1)
        finally:
            environment.close()

    def report(self) -> ControlReport:
        """Expose a selected policy only after a scheduled validation has occurred."""

        if not self.selector.history:
            raise ValueError("no validation checkpoint is available yet")
        return ControlReport(
            self.selector.selected(),
            tuple(self.returns),
            tuple(self.diagnostics),
            tuple(self.selector.history),
            self.selector.best_episode,
            self.environment_steps,
        )


def train_dqn(config: ControlConfig) -> ControlReport:
    """Execute one complete declared DQN budget; test evaluation remains external."""

    session = DQNSession(config)
    session.advance(config.episodes)
    return session.report()
