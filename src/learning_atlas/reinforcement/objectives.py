"""Explicit policy-gradient, PPO-Clip, GAE, and DQN mathematics.

No RL reference library supplies these updates. All targets are detached and
validated before optimization; log-space probabilities avoid softmax/log loss.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learning_atlas.deep.training import Batch
from learning_atlas.reinforcement.validation import FloatArray, integer, scalar, vector


def generalized_advantage(
    rewards: FloatArray,
    values: FloatArray,
    next_values: FloatArray,
    terminated: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    boundaries: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """O(T) GAE: termination masks bootstrap, every episode boundary masks carry.

    At a time limit use the value of the physical final observation, never the
    reset observation. A rollout cut also stops carry but preserves bootstrap.
    """

    r = vector(rewards, "rewards")
    v = vector(values, "values", size=len(r))
    nv = vector(next_values, "next_values", size=len(r))
    scalar(gamma, "gamma", 0.0, 1.0)
    scalar(gae_lambda, "gae_lambda", 0.0, 1.0)
    for name, flags in (("terminated", terminated), ("boundaries", boundaries)):
        if flags.shape != r.shape or flags.dtype != np.bool_:
            raise ValueError(f"{name} must be aligned bool flags")
    if np.any(terminated & ~boundaries):
        raise ValueError("every termination must also be an episode boundary")
    delta = r + gamma * (~terminated) * nv - v
    advantages = np.empty_like(r)
    carry = 0.0
    for index in range(len(r) - 1, -1, -1):
        carry = float(delta[index]) + gamma * gae_lambda * (not boundaries[index]) * carry
        advantages[index] = carry
    if not np.isfinite(advantages).all():
        raise ValueError("GAE overflowed; rescale rewards or values")
    return advantages, advantages + v


def checked_tensor(tensor: torch.Tensor, name: str, *, rank: int, length: int) -> None:
    """Validate bounded CPU tensor shape and finiteness before semantic use."""

    if (
        tensor.device.type != "cpu"
        or tensor.ndim != rank
        or tensor.shape[0] != length
        or not 1 <= tensor.numel() <= 2_000_000
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError(f"{name} requires finite aligned bounded CPU tensors")


def policy_statistics(
    logits: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable categorical log probabilities and entropy for chosen actions."""

    if logits.ndim != 2 or not 2 <= logits.shape[1] <= 16:
        raise ValueError("policy logits require [batch,2..16 actions]")
    checked_tensor(logits, "logits", rank=2, length=len(logits))
    checked_tensor(actions, "actions", rank=1, length=len(logits))
    if actions.dtype != torch.int64 or bool(((actions < 0) | (actions >= logits.shape[1])).any()):
        raise ValueError("actions must be valid int64 categorical indices")
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(1, actions[:, None]).squeeze(1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=1)
    return chosen, entropy


def clipped_surrogate(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO pessimistic objective, including negative advantages and detached rollout data."""

    scalar(clip_ratio, "clip_ratio", 1e-6, 0.999)
    if log_probs.ndim != 1:
        raise ValueError("log_probs must be rank one")
    for name, tensor in (
        ("log_probs", log_probs),
        ("old_log_probs", old_log_probs),
        ("advantages", advantages),
    ):
        checked_tensor(tensor, name, rank=1, length=len(log_probs))
    log_ratio = log_probs - old_log_probs.detach()
    if bool((log_ratio.detach().abs() > 60).any()):
        raise ValueError("policy likelihood ratio exceeds the safe numeric range")
    ratio = log_ratio.exp()
    adv = advantages.detach()
    loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv).mean()
    with torch.no_grad():
        metrics = {
            "approximate_kl": float(((ratio - 1) - log_ratio).mean()),
            "clip_fraction": float(((ratio - 1).abs() > clip_ratio).float().mean()),
        }
    return loss, metrics


class ActorCritic(nn.Module):
    """Separate actor and critic avoid leaking value gradients into the policy baseline."""

    def __init__(self, observations: int = 4, actions: int = 2, hidden: int = 64) -> None:
        super().__init__()
        integer(observations, "observations", 1, 128)
        integer(actions, "actions", 2, 16)
        integer(hidden, "hidden", 4, 256)
        self.actor = nn.Sequential(
            nn.Linear(observations, hidden), nn.Tanh(), nn.Linear(hidden, actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(observations, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Concatenate categorical logits and scalar state values for the shared Trainer."""

        return torch.cat((self.actor(observations), self.critic(observations)), dim=1)


@dataclass(slots=True)
class PolicyObjective:
    """Batch=(observations,actions,advantages,returns,old_log_probs)."""

    ppo: bool
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    diagnostics: dict[str, float] = field(default_factory=dict, init=False)

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Joint actor/critic loss, with no gradient through fitted rollout targets."""

        if len(batch) != 5:
            raise ValueError("policy objective requires five aligned tensors")
        observations, actions, advantages, returns, old_log_probs = batch
        if observations.ndim != 2:
            raise ValueError("observations require a matrix")
        checked_tensor(observations, "observations", rank=2, length=len(observations))
        output = model(observations)
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] < 3:
            raise ValueError("actor-critic must return logits followed by one value")
        log_probs, entropy = policy_statistics(output[:, :-1], actions)
        for name, tensor in (
            ("advantages", advantages),
            ("returns", returns),
            ("old_log_probs", old_log_probs),
        ):
            checked_tensor(tensor, name, rank=1, length=len(observations))
        if self.ppo:
            policy_loss, metrics = clipped_surrogate(
                log_probs, old_log_probs, advantages, self.clip_ratio
            )
        else:
            policy_loss = -(log_probs * advantages.detach()).mean()
            metrics = {"approximate_kl": 0.0, "clip_fraction": 0.0}
        value_loss = F.mse_loss(output[:, -1], returns.detach())
        scalar(self.entropy_coefficient, "entropy_coefficient", 0.0, 1.0)
        scalar(self.value_coefficient, "value_coefficient", 0.0, 10.0)
        self.diagnostics = {
            **metrics,
            "entropy": float(entropy.detach().mean()),
            "value_mse": float(value_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
        }
        return (
            policy_loss
            + self.value_coefficient * value_loss
            - self.entropy_coefficient * entropy.mean()
        )


@dataclass(frozen=True, slots=True)
class DQNObjective:
    """Huber TD regression; replay targets are computed from a detached target network."""

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Batch=(observations,actions,targets); terminal masking happens in target construction."""

        if len(batch) != 3:
            raise ValueError("DQN objective requires three aligned tensors")
        observations, actions, targets = batch
        if observations.ndim != 2:
            raise ValueError("observations require a matrix")
        checked_tensor(observations, "observations", rank=2, length=len(observations))
        output = model(observations)
        if not isinstance(output, torch.Tensor):
            raise ValueError("Q network must return a tensor")
        policy_statistics(output, actions)  # Shared categorical-index and finite-output validation.
        checked_tensor(targets, "targets", rank=1, length=len(observations))
        chosen = output.gather(1, actions[:, None]).squeeze(1)
        return F.smooth_l1_loss(chosen, targets.detach())
