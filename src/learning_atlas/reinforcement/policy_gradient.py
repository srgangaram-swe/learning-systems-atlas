"""Episode-aligned REINFORCE and PPO interaction loops using the shared Trainer."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from learning_atlas.core.reproducibility import derive_seed
from learning_atlas.deep.training import Batch
from learning_atlas.reinforcement.control import (
    ControlReport,
    ValidationSelector,
    action_for,
    create_model,
    create_trainer,
    deterministic_cpu,
    observation_tensor,
)
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.objectives import (
    PolicyObjective,
    generalized_advantage,
    policy_statistics,
)


@dataclass(slots=True)
class Rollout:
    """Owned on-policy transitions; complete episodes prevent reset-observation bootstrap."""

    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    boundaries: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)

    def batch(self, config: ControlConfig) -> Batch:
        """Freeze rollout targets once; PPO reuses precisely these old-policy quantities."""

        if not self.rewards:
            raise ValueError("cannot optimize an empty rollout")
        rewards, values, next_values = map(
            np.asarray, (self.rewards, self.values, self.next_values)
        )
        term, boundaries = np.asarray(self.terminated), np.asarray(self.boundaries)
        if config.method == "reinforce":
            # lambda=1 telescopes to Monte Carlo reward-to-go, with critic
            # bootstrap only at external time limits, not true terminations.
            _, returns = generalized_advantage(
                rewards, values, next_values, term, boundaries, gamma=config.gamma, gae_lambda=1.0
            )
            advantages = returns - values if config.learned_baseline else returns.copy()
        else:
            advantages, returns = generalized_advantage(
                rewards,
                values,
                next_values,
                term,
                boundaries,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
        # Fixed within the rollout and across optimization epochs, not normalized
        # per episode (which would erase relative return information).
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        return (
            torch.stack(self.observations),
            torch.tensor(self.actions, dtype=torch.int64),
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
        )


def collect_episode(
    environment: gym.Env[np.ndarray, int],
    model: nn.Module,
    config: ControlConfig,
    *,
    episode: int,
    rng: np.random.Generator,
    rollout: Rollout,
) -> float:
    """Collect one bounded trajectory under the frozen current policy."""

    observation, _ = environment.reset(seed=derive_seed(config.seed, stream=1000 + episode))
    total = 0.0
    for _ in range(config.max_steps):
        tensor = observation_tensor(observation)
        action = action_for(model, observation, rng, stochastic=True)
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        with torch.no_grad():
            output = model(tensor[None, :])
            next_output = model(observation_tensor(next_observation)[None, :])
            chosen, _ = policy_statistics(output[:, :2], torch.tensor([action]))
        rollout.observations.append(tensor)
        rollout.actions.append(action)
        rollout.rewards.append(float(reward) * config.reward_scale)
        rollout.values.append(float(output[0, -1]))
        rollout.next_values.append(float(next_output[0, -1]))
        rollout.terminated.append(bool(terminated))
        rollout.boundaries.append(bool(terminated or truncated))
        rollout.log_probs.append(float(chosen[0]))
        total += float(reward)
        observation = next_observation
        if terminated or truncated:
            break
    return total


def train_policy_gradient(config: ControlConfig) -> ControlReport:
    """Train entirely offline; select a checkpoint on validation before test access.

    PPO rolls out a bounded number of whole episodes, a deliberate CPU reference
    tradeoff rather than a fixed-length vectorized actor implementation. The
    maximum rollout is rollout_episodes * max_steps transitions.
    """

    if config.method not in {"reinforce", "ppo"}:
        raise ValueError("policy-gradient training requires reinforce or ppo")
    with deterministic_cpu(config.seed):
        return _train_owned(config)


def _train_owned(config: ControlConfig) -> ControlReport:
    model = create_model(config)
    trainer = create_trainer(config)
    optimizer = trainer.create_optimizer(model)
    selector = ValidationSelector(config, model)
    environment = gym.make("CartPole-v1", max_episode_steps=config.max_steps)
    action_rng = np.random.default_rng(derive_seed(config.seed, stream=2))
    batch_rng = np.random.default_rng(derive_seed(config.seed, stream=3))
    objective = PolicyObjective(
        ppo=config.method == "ppo",
        clip_ratio=config.clip_ratio,
        entropy_coefficient=config.entropy_coefficient,
        value_coefficient=config.value_coefficient,
    )
    returns: list[float] = []
    diagnostics: list[dict[str, float]] = []
    environment_steps, updates = 0, 0
    try:
        for first in range(0, config.episodes, config.rollout_episodes):
            rollout = Rollout()
            stop = min(first + config.rollout_episodes, config.episodes)
            for episode in range(first, stop):
                returns.append(
                    collect_episode(
                        environment, model, config, episode=episode, rng=action_rng, rollout=rollout
                    )
                )
            batch = rollout.batch(config)
            environment_steps += len(rollout.rewards)
            epochs = config.optimization_epochs if config.method == "ppo" else 1
            last_metrics: dict[str, float] = {}
            early_stop = False
            for _ in range(epochs):
                indices = batch_rng.permutation(len(batch[0]))
                # REINFORCE uses one whole-rollout policy update. Reusing its
                # old advantages in shuffled repeated minibatches would silently
                # turn it into an uncorrected off-policy update.
                size = config.batch_size if config.method == "ppo" else len(indices)
                for offset in range(0, len(indices), size):
                    mini = tuple(tensor[indices[offset : offset + size]] for tensor in batch)
                    if config.method == "ppo":
                        with torch.no_grad():
                            objective.loss(model, mini)
                        if objective.diagnostics["approximate_kl"] > config.target_kl:
                            early_stop = True
                            break
                    updates += 1
                    loss = trainer.train_batch(model, optimizer, mini, objective, step=updates)
                    gradient_norm = float(
                        torch.sqrt(
                            sum(
                                (
                                    (parameter.grad.detach() ** 2).sum()
                                    for parameter in model.parameters()
                                    if parameter.grad is not None
                                ),
                                start=torch.tensor(0.0),
                            )
                        )
                    )
                    last_metrics = {
                        **objective.diagnostics,
                        "loss": loss,
                        "gradient_norm": gradient_norm,
                    }
                if early_stop:
                    break
            diagnostics.append(
                {
                    **last_metrics,
                    "episode": float(stop),
                    "environment_steps": float(environment_steps),
                    "optimizer_updates": float(updates),
                    "kl_early_stop": float(early_stop),
                }
            )
            if (
                stop // config.validation_interval != first // config.validation_interval
                or stop == config.episodes
            ):
                selector.consider(model, stop)
    finally:
        environment.close()
    return ControlReport(
        selector.selected(),
        tuple(returns),
        tuple(diagnostics),
        tuple(selector.history),
        selector.best_episode,
        environment_steps,
    )
