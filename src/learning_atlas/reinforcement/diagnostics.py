"""Independent policy-gradient variance diagnostics, never used for model selection."""

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from learning_atlas.core.reproducibility import derive_seed
from learning_atlas.reinforcement.control import deterministic_cpu
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.objectives import generalized_advantage, policy_statistics
from learning_atlas.reinforcement.policy_gradient import Rollout, collect_episode
from learning_atlas.reinforcement.validation import integer


def baseline_variance(
    model: nn.Module, config: ControlConfig, *, episodes: int = 32
) -> dict[str, float]:
    """Estimate trace(Cov[trajectory score-gradient]) on fresh frozen-policy rollouts.

    Pair identical trajectories with/without the already learned state baseline.
    This estimates variance, not squared gradient norm or training loss. It is
    a diagnostic of this fitted policy, not a theorem that every learned baseline
    reduces variance. No diagnostic observation is used to fit the critic.
    """

    integer(episodes, "variance episodes", 2, 256)
    environment = gym.make("CartPole-v1", max_episode_steps=config.max_steps)
    probe = config.model_copy(update={"seed": derive_seed(config.seed, stream=400)})
    rng = np.random.default_rng(derive_seed(config.seed, stream=401))
    gradients: dict[str, list[np.ndarray]] = {"without_baseline": [], "learned_baseline": []}
    parameters = tuple(model.parameters())
    was_training = model.training
    model.eval()
    try:
        with deterministic_cpu(probe.seed):
            for episode in range(episodes):
                rollout = Rollout()
                collect_episode(
                    environment, model, probe, episode=episode, rng=rng, rollout=rollout
                )
                values = np.asarray(rollout.values)
                _, returns = generalized_advantage(
                    np.asarray(rollout.rewards),
                    values,
                    np.asarray(rollout.next_values),
                    np.asarray(rollout.terminated),
                    np.asarray(rollout.boundaries),
                    gamma=config.gamma,
                    gae_lambda=1.0,
                )
                discounts = config.gamma ** np.arange(len(returns))
                for name, baseline in (
                    ("without_baseline", np.zeros_like(values)),
                    ("learned_baseline", values),
                ):
                    output = model(torch.stack(rollout.observations))
                    log_probs, _ = policy_statistics(output[:, :2], torch.tensor(rollout.actions))
                    weights = torch.tensor(discounts * (returns - baseline), dtype=torch.float32)
                    objective = -(log_probs * weights).sum()
                    gradient = torch.autograd.grad(objective, parameters, allow_unused=True)
                    flattened = torch.cat(
                        [
                            (torch.zeros_like(parameter) if item is None else item).reshape(-1)
                            for parameter, item in zip(parameters, gradient, strict=True)
                        ]
                    )
                    gradients[name].append(flattened.detach().double().numpy())
    finally:
        environment.close()
        model.train(was_training)
    variances = {
        name: float(np.var(np.stack(samples), axis=0, ddof=1).sum())
        for name, samples in gradients.items()
    }
    denominator = variances["without_baseline"]
    return {
        **variances,
        "variance_ratio": variances["learned_baseline"] / denominator if denominator > 0 else 0.0,
        "ratio_defined": float(denominator > 0),
        "independent_trajectories": float(episodes),
    }
