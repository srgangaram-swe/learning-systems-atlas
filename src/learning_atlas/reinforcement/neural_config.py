"""Strict CPU-only control budgets and optimizer settings."""

from typing import Literal, Self

from pydantic import Field, model_validator

from learning_atlas.core.config import StrictConfig


class ControlConfig(StrictConfig):
    """One predeclared control run; episode count may grow only on checkpoint resume."""

    method: Literal["reinforce", "dqn", "ppo"]
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    episodes: int = Field(default=500, ge=1, le=10_000)
    max_steps: int = Field(default=500, ge=1, le=500)
    hidden_units: int = Field(default=64, ge=4, le=256)
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1)
    gamma: float = Field(default=0.99, ge=0.0, lt=1.0)
    reward_scale: float = Field(default=0.01, gt=0.0, le=1.0)
    grad_clip: float = Field(default=0.5, gt=0.0, le=100.0)
    batch_size: int = Field(default=64, ge=2, le=2048)
    rollout_episodes: int = Field(default=8, ge=1, le=128)
    optimization_epochs: int = Field(default=8, ge=1, le=64)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    target_kl: float = Field(default=0.03, gt=0.0, le=1.0)
    entropy_coefficient: float = Field(default=0.01, ge=0.0, le=1.0)
    value_coefficient: float = Field(default=0.5, gt=0.0, le=10.0)
    learned_baseline: bool = True
    replay_capacity: int = Field(default=20_000, ge=2, le=100_000)
    learning_starts: int = Field(default=256, ge=2, le=100_000)
    train_frequency: int = Field(default=4, ge=1, le=128)
    target_interval: int = Field(default=250, ge=1, le=100_000)
    replay_enabled: bool = True
    target_enabled: bool = True
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_steps: int = Field(default=20_000, ge=1, le=1_000_000)
    validation_interval: int = Field(default=25, ge=1, le=10_000)
    validation_episodes: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def enforce_resource_and_schedule_contract(self) -> Self:
        """Fail before allocation or interaction for incompatible coupled settings."""

        if self.episodes * self.max_steps > 2_000_000:
            raise ValueError("control budget exceeds two million transitions")
        if self.batch_size > self.replay_capacity or self.learning_starts > self.replay_capacity:
            raise ValueError("replay capacity must cover batch_size and learning_starts")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot exceed epsilon_start")
        return self
