"""Bounded, non-executable JSON checkpoints for episode-aligned DQN recovery.

No pickle, arbitrary imports, object hooks, or external tensor paths. A checksum
detects accidental corruption, not authenticity. Loading constructs a new session
and exposes it only after complete validation; caller-owned sessions are untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.reinforcement.dqn import DQNSession
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.validation import ReinforcementError

_MAX_BYTES = 32 * 1024 * 1024
FiniteValues = Annotated[list[FiniteFloat], Field(max_length=1_000_000)]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TensorRecord(_Record):
    """Float32 model/Adam tensor, validated before allocation."""

    shape: list[Annotated[int, Field(ge=1, le=1_000_000)]] = Field(max_length=4)
    values: FiniteValues

    @model_validator(mode="after")
    def match_shape(self) -> Self:
        if math.prod(self.shape) != len(self.values):
            raise ValueError("tensor shape does not match values")
        if any(abs(value) > float(np.finfo(np.float32).max) for value in self.values):
            raise ValueError("tensor exceeds float32 range")
        return self

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> TensorRecord:
        if tensor.dtype != torch.float32:
            raise ReinforcementError("DQN checkpoints support float32 model/Adam tensors only")
        return cls(shape=list(tensor.shape), values=tensor.detach().cpu().reshape(-1).tolist())

    def tensor(self) -> torch.Tensor:
        return torch.tensor(self.values, dtype=torch.float32).reshape(self.shape)


class AdamRecord(_Record):
    step: TensorRecord
    exp_avg: TensorRecord
    exp_avg_sq: TensorRecord


class RandomRecord(_Record):
    state: int = Field(ge=0, lt=2**128)
    inc: int = Field(ge=0, lt=2**128)
    has_uint32: int = Field(ge=0, le=1)
    uinteger: int = Field(ge=0, lt=2**32)

    @classmethod
    def capture(cls, rng: np.random.Generator) -> RandomRecord:
        raw = rng.bit_generator.state
        return cls(
            state=raw["state"]["state"],
            inc=raw["state"]["inc"],
            has_uint32=raw["has_uint32"],
            uinteger=raw["uinteger"],
        )

    def restore(self, rng: np.random.Generator) -> None:
        rng.bit_generator.state = {
            "bit_generator": "PCG64",
            "state": {"state": self.state, "inc": self.inc},
            "has_uint32": self.has_uint32,
            "uinteger": self.uinteger,
        }


class ReplayRecord(_Record):
    observations: list[Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]] = Field(
        max_length=100_000
    )
    next_observations: list[Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]] = (
        Field(max_length=100_000)
    )
    actions: list[Annotated[int, Field(ge=0, le=1)]] = Field(max_length=100_000)
    rewards: FiniteValues
    terminated: list[bool] = Field(max_length=100_000)
    position: int = Field(ge=0, lt=100_000)


class DQNCheckpoint(_Record):
    """All resume-critical state, including selection, replay order, and separate RNGs."""

    config: ControlConfig
    runtime: dict[str, str]
    model: dict[str, TensorRecord] = Field(max_length=32)
    target: dict[str, TensorRecord] = Field(max_length=32)
    selected: dict[str, TensorRecord] = Field(max_length=32)
    optimizer: list[AdamRecord] = Field(max_length=32)
    replay: ReplayRecord
    action_rng: RandomRecord
    replay_rng: RandomRecord
    returns: FiniteValues
    diagnostics: list[dict[str, FiniteFloat]] = Field(max_length=10_000)
    validation: list[dict[str, FiniteFloat]] = Field(max_length=10_000)
    selected_episode: int = Field(ge=0, le=10_000)
    best_score: FiniteFloat
    environment_steps: int = Field(ge=0, le=2_000_000)
    updates: int = Field(ge=0, le=2_000_000)


class Envelope(_Record):
    schema_version: Literal["learning-atlas-dqn/1"] = "learning-atlas-dqn/1"
    sha256: str = Field(pattern="^[0-9a-f]{64}$")
    payload: DQNCheckpoint


def _digest(payload: DQNCheckpoint) -> str:
    encoded = json.dumps(
        payload.model_dump(mode="json"), sort_keys=True, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _runtime() -> dict[str, str]:
    """Exact-resume compatibility, not a cross-version checkpoint migration promise."""

    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        **{
            name: version(name)
            for name in ("learning-systems-atlas", "torch", "numpy", "gymnasium")
        },
    }


def save_dqn(session: DQNSession, store: ArtifactStore, path: str = "dqn-checkpoint.json") -> Path:
    """Atomically save a completed episode boundary; never include unused replay capacity."""

    replay = session.replay
    optimizer = []
    for parameter in session.model.parameters():
        state = session.optimizer.state.get(parameter)
        if state:
            optimizer.append(
                AdamRecord(
                    **{
                        key: TensorRecord.capture(state[key])
                        for key in ("step", "exp_avg", "exp_avg_sq")
                    }
                )
            )
    payload = DQNCheckpoint(
        config=session.config,
        runtime=_runtime(),
        model={
            key: TensorRecord.capture(value) for key, value in session.model.state_dict().items()
        },
        target={
            key: TensorRecord.capture(value) for key, value in session.target.state_dict().items()
        },
        selected={
            key: TensorRecord.capture(value) for key, value in session.selector.best_state.items()
        },
        optimizer=optimizer,
        replay=ReplayRecord(
            observations=replay.observations[: replay.size].tolist(),
            next_observations=replay.next_observations[: replay.size].tolist(),
            actions=replay.actions[: replay.size].tolist(),
            rewards=replay.rewards[: replay.size].tolist(),
            terminated=replay.terminated[: replay.size].tolist(),
            position=replay.position,
        ),
        action_rng=RandomRecord.capture(session.action_rng),
        replay_rng=RandomRecord.capture(session.replay_rng),
        returns=session.returns,
        diagnostics=session.diagnostics,
        validation=session.selector.history,
        selected_episode=session.selector.best_episode,
        best_score=session.selector.best_score,
        environment_steps=session.environment_steps,
        updates=session.updates,
    )
    encoded = Envelope(sha256=_digest(payload), payload=payload).model_dump_json(indent=2)
    if len(encoded.encode()) > _MAX_BYTES:
        raise ReinforcementError("DQN checkpoint exceeds 32 MiB storage budget")
    return store.write_text(path, encoded + "\n")


def _model_state(
    records: dict[str, TensorRecord], expected: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if records.keys() != expected.keys():
        raise ValueError("checkpoint model tensor names are incompatible")
    result = {}
    for key, record in records.items():
        if record.shape != list(expected[key].shape):
            raise ValueError("checkpoint model tensor shape is incompatible")
        result[key] = record.tensor()
    return result


def _restore(payload: DQNCheckpoint, expected: ControlConfig) -> DQNSession:
    if payload.config != expected:
        raise ValueError("checkpoint configuration does not match the requested run")
    if payload.runtime != _runtime():
        raise ValueError("checkpoint runtime does not match this platform and dependency set")
    session = DQNSession(expected)
    episodes = len(payload.returns)
    if episodes > expected.episodes or len(payload.diagnostics) != episodes:
        raise ValueError("checkpoint episode history is inconsistent")
    if not episodes <= payload.environment_steps <= episodes * expected.max_steps:
        raise ValueError("checkpoint interaction count is inconsistent")
    if (
        any(
            value != int(value) or not 1 <= value <= expected.max_steps for value in payload.returns
        )
        or sum(payload.returns) != payload.environment_steps
    ):
        raise ValueError("checkpoint CartPole returns do not match interaction counts")
    first_update = max(expected.learning_starts, expected.batch_size)
    expected_updates = max(
        0,
        payload.environment_steps // expected.train_frequency
        - (first_update - 1) // expected.train_frequency,
    )
    if payload.updates != expected_updates:
        raise ValueError("checkpoint update count is inconsistent")
    session.model.load_state_dict(_model_state(payload.model, dict(session.model.state_dict())))
    session.target.load_state_dict(_model_state(payload.target, dict(session.target.state_dict())))
    session.selector.best_state = _model_state(payload.selected, dict(session.model.state_dict()))
    parameters = list(session.model.parameters())
    if len(payload.optimizer) != (len(parameters) if payload.updates else 0):
        raise ValueError("checkpoint optimizer parameter count is inconsistent")
    for parameter, record in zip(parameters, payload.optimizer, strict=bool(payload.updates)):
        if record.exp_avg.shape != list(parameter.shape) or record.exp_avg_sq.shape != list(
            parameter.shape
        ):
            raise ValueError("checkpoint Adam tensor shape is incompatible")
        step = record.step.tensor()
        square = record.exp_avg_sq.tensor()
        if step.ndim != 0 or float(step) != payload.updates or bool((square < 0).any()):
            raise ValueError("checkpoint Adam step or variance is invalid")
        session.optimizer.state[parameter] = {
            "step": step,
            "exp_avg": record.exp_avg.tensor(),
            "exp_avg_sq": square,
        }
    replay = payload.replay
    size = len(replay.actions)
    if (
        size != min(payload.environment_steps, expected.replay_capacity)
        or replay.position != payload.environment_steps % expected.replay_capacity
        or any(
            len(values) != size
            for values in (
                replay.observations,
                replay.next_observations,
                replay.rewards,
                replay.terminated,
            )
        )
    ):
        raise ValueError("checkpoint replay lengths/position are inconsistent")
    for observation, action, reward, next_observation, terminated in zip(
        replay.observations,
        replay.actions,
        replay.rewards,
        replay.next_observations,
        replay.terminated,
        strict=True,
    ):
        session.replay.append(
            np.asarray(observation), action, reward, np.asarray(next_observation), terminated
        )
    session.replay.position = replay.position
    payload.action_rng.restore(session.action_rng)
    payload.replay_rng.restore(session.replay_rng)
    if payload.selected_episode > episodes or (
        not payload.validation and payload.selected_episode != 0
    ):
        raise ValueError("checkpoint selection episode is inconsistent")
    expected_validation = [
        index
        for index in range(1, episodes + 1)
        if index % expected.validation_interval == 0 or index == expected.episodes
    ]
    if len(payload.validation) != len(expected_validation):
        raise ValueError("checkpoint validation schedule is inconsistent")
    for validation_record, episode in zip(payload.validation, expected_validation, strict=True):
        if (
            validation_record.keys() != {"episode", "mean_return"}
            or validation_record["episode"] != episode
            or not 1 <= validation_record["mean_return"] <= expected.max_steps
        ):
            raise ValueError("checkpoint validation record is invalid")
    if payload.validation:
        best = max(payload.validation, key=lambda record: record["mean_return"])
        if payload.best_score != best["mean_return"] or payload.selected_episode != best["episode"]:
            raise ValueError("checkpoint selected policy is inconsistent with validation")
    elif payload.best_score != -1.0:
        raise ValueError("checkpoint unvalidated selection score is invalid")
    session.returns = list(payload.returns)
    session.diagnostics = list(payload.diagnostics)
    session.environment_steps, session.updates = payload.environment_steps, payload.updates
    session.selector.best_episode, session.selector.best_score = (
        payload.selected_episode,
        payload.best_score,
    )
    session.selector.history = list(payload.validation)
    return session


def load_dqn(path: Path, expected: ControlConfig) -> DQNSession:
    """Fail closed on oversized, malformed, corrupt, or incompatible state.

    Reads at most 32 MiB + 1 byte even if the file changes after stat. Only the
    newly allocated session is mutated; no external state is restored on failure.
    """

    try:
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("checkpoint exceeds 32 MiB storage budget")
        with path.open("rb") as stream:
            encoded = stream.read(_MAX_BYTES + 1)
        if len(encoded) > _MAX_BYTES:
            raise ValueError("checkpoint exceeds 32 MiB storage budget")
        envelope = Envelope.model_validate_json(encoded)
        if _digest(envelope.payload) != envelope.sha256:
            raise ValueError("checkpoint checksum mismatch")
        return _restore(envelope.payload, expected)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError) as error:
        raise ReinforcementError(
            "DQN checkpoint is unreadable, corrupt, or incompatible"
        ) from error
