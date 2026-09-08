"""Bounded uniform replay with owned observations and O(1) ring insertion."""

import numpy as np
import torch

from learning_atlas.deep.training import Batch
from learning_atlas.reinforcement.control import observation_tensor
from learning_atlas.reinforcement.validation import FloatArray, integer, scalar


class ReplayBuffer:
    """Fixed-capacity structure-of-arrays buffer; sampling is uniform without replacement."""

    def __init__(self, capacity: int) -> None:
        self.capacity = integer(capacity, "capacity", 2, 100_000)
        self.observations = np.zeros((capacity, 4), dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.size = 0
        self.position = 0

    def append(
        self,
        observation: FloatArray,
        action: int,
        reward: float,
        next_observation: FloatArray,
        terminated: bool,
    ) -> None:
        """Validate all fields before committing one transition; truncation is not termination."""

        first = observation_tensor(observation).numpy()
        second = observation_tensor(next_observation).numpy()
        integer(action, "action", 0, 1)
        scalar(reward, "reward", -1e6, 1e6)
        if not isinstance(terminated, bool):
            raise ValueError("terminated must be bool")
        index = self.position
        self.observations[index], self.next_observations[index] = first, second
        self.actions[index], self.rewards[index], self.terminated[index] = (
            action,
            reward,
            terminated,
        )
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, rng: np.random.Generator, *, uniform: bool = True) -> Batch:
        """Copy a uniform minibatch, or a chronological recent-window ablation.

        The no-replay ablation retains the same minibatch size and optimizer
        cadence; it changes sampling age/correlation, not the update budget.
        """

        integer(count, "sample count", 1, self.size)
        indices = (
            rng.choice(self.size, count, replace=False)
            if uniform
            else (np.arange(self.position - count, self.position) % self.capacity)
        )
        return tuple(
            torch.from_numpy(array[indices].copy())
            for array in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.terminated,
            )
        )
