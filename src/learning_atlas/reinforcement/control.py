"""CartPole adapters, owned random streams, and independent policy evaluation."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from learning_atlas.core.reproducibility import derive_seed
from learning_atlas.deep.training import Trainer, TrainerConfig
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.objectives import ActorCritic
from learning_atlas.reinforcement.validation import FloatArray, integer, vector


@contextmanager
def deterministic_cpu(seed: int) -> Iterator[None]:
    """Restore Torch global state even on failure; NumPy streams are locally owned.

    Reference execution is single-threaded. This context is intentionally not a
    concurrent training API because Torch determinism/thread settings are global.
    """

    integer(seed, "seed", 0, 2**32 - 1)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
            yield
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)
            torch.set_num_threads(threads)


def create_model(config: ControlConfig) -> nn.Module:
    """Initialize a CPU network without advancing the caller's Torch random state."""

    with deterministic_cpu(derive_seed(config.seed, stream=1)):
        if config.method == "dqn":
            return nn.Sequential(
                nn.Linear(4, config.hidden_units),
                nn.ReLU(),
                nn.Linear(config.hidden_units, config.hidden_units),
                nn.ReLU(),
                nn.Linear(config.hidden_units, 2),
            )
        return ActorCritic(hidden=config.hidden_units)


def create_trainer(config: ControlConfig) -> Trainer:
    """Use the existing Trainer's finite-loss/gradient/optimizer safety boundary."""

    return Trainer(
        TrainerConfig(
            max_epochs=1,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            grad_clip_norm=config.grad_clip,
            device="cpu",
        )
    )


def observation_tensor(observation: FloatArray) -> torch.Tensor:
    """Validate physical observations before conversion to bounded float32 input."""

    values = vector(observation, "CartPole observation", size=4)
    if np.max(np.abs(values)) > 1e6:
        raise ValueError("CartPole observation exceeds the numeric budget")
    return torch.tensor(values, dtype=torch.float32)


def action_for(
    model: nn.Module,
    observation: FloatArray,
    rng: np.random.Generator,
    *,
    stochastic: bool,
) -> int:
    """Choose categorical actions with an explicit NumPy stream, never global sampling."""

    with torch.no_grad():
        output = model(observation_tensor(observation)[None, :])
        if not isinstance(output, torch.Tensor) or output.shape not in {(1, 2), (1, 3)}:
            raise ValueError("control model must return two actions and an optional value")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("control model produced non-finite output")
        logits = output[0, :2]
        if not stochastic:
            return int(logits.argmax())
        probabilities = logits.softmax(0).double().numpy()
        probabilities /= probabilities.sum()
        return int(rng.choice(2, p=probabilities))


def evaluate_control(
    model: nn.Module | None,
    *,
    seed: int,
    episodes: int,
    max_steps: int = 500,
    stochastic: bool = False,
) -> FloatArray:
    """Independent environment/action streams; no model updates or training RNG reads."""

    integer(seed, "seed", 0, 2**32 - 1)
    integer(episodes, "episodes", 1, 1000)
    integer(max_steps, "max_steps", 1, 500)
    environment = gym.make("CartPole-v1", max_episode_steps=max_steps)
    rng = np.random.default_rng(derive_seed(seed, stream=1))
    returns = np.zeros(episodes)
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=derive_seed(seed, stream=100 + episode))
            for _ in range(max_steps):
                action = (
                    int(rng.integers(2))
                    if model is None
                    else action_for(model, observation, rng, stochastic=stochastic)
                )
                observation, reward, terminated, truncated, _ = environment.step(action)
                returns[episode] += float(reward)
                if terminated or truncated:
                    break
    finally:
        environment.close()
        if model is not None:
            model.train(was_training)
    return returns


@dataclass(frozen=True, slots=True)
class ControlReport:
    """Training evidence and training-validation-selected policy, before test access."""

    model: nn.Module
    returns: tuple[float, ...]
    diagnostics: tuple[dict[str, float], ...]
    validation: tuple[dict[str, float], ...]
    selected_episode: int
    environment_steps: int


class ValidationSelector:
    """Select only on a fixed training-validation stream; ties prefer earlier policy."""

    def __init__(self, config: ControlConfig, model: nn.Module) -> None:
        self.config = config
        self.best_score = -1.0
        self.best_episode = 0
        self.best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.history: list[dict[str, float]] = []

    def consider(self, model: nn.Module, episode: int) -> None:
        """Validation never uses the benchmark's untouched-test stream."""

        returns = evaluate_control(
            model,
            seed=derive_seed(self.config.seed, stream=20),
            episodes=self.config.validation_episodes,
            max_steps=self.config.max_steps,
        )
        score = float(returns.mean())
        self.history.append({"episode": float(episode), "mean_return": score})
        if score > self.best_score:
            self.best_score, self.best_episode = score, episode
            self.best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    def selected(self) -> nn.Module:
        """Construct a separate selected model without mutating continuing training state."""

        model = create_model(self.config)
        model.load_state_dict(self.best_state)
        return model
