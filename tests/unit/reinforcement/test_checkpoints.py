"""DQN continuation, replay/optimizer integrity, and hostile checkpoint rejection."""

import hashlib
import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.reinforcement.checkpoints import TensorRecord, load_dqn, save_dqn
from learning_atlas.reinforcement.dqn import DQNSession
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.validation import ReinforcementError

pytestmark = pytest.mark.unit


def small_config(**kwargs):
    return ControlConfig(
        method="dqn",
        episodes=12,
        max_steps=30,
        hidden_units=8,
        batch_size=8,
        learning_starts=8,
        replay_capacity=32,
        target_interval=2,
        validation_interval=3,
        validation_episodes=2,
        **kwargs,
    )


def test_checkpoint_resume_matches_uninterrupted_training_bit_for_bit(tmp_path):
    config = small_config()
    uninterrupted = DQNSession(config)
    uninterrupted.advance(12)
    interrupted = DQNSession(config)
    interrupted.advance(5)
    checkpoint = save_dqn(interrupted, ArtifactStore(tmp_path))
    restored = load_dqn(checkpoint, config)
    restored.advance(12)
    assert restored.returns == uninterrupted.returns
    assert restored.diagnostics == uninterrupted.diagnostics
    assert restored.selector.history == uninterrupted.selector.history
    assert restored.selector.best_episode == uninterrupted.selector.best_episode
    assert restored.epsilon == uninterrupted.epsilon
    assert restored.action_rng.bit_generator.state == uninterrupted.action_rng.bit_generator.state
    assert restored.replay_rng.bit_generator.state == uninterrupted.replay_rng.bit_generator.state
    for network in ("model", "target"):
        for key, tensor in getattr(uninterrupted, network).state_dict().items():
            assert torch.equal(tensor, getattr(restored, network).state_dict()[key])
    np.testing.assert_array_equal(restored.replay.observations, uninterrupted.replay.observations)
    assert restored.report().environment_steps == uninterrupted.report().environment_steps


def test_empty_session_roundtrip_and_no_unvalidated_report(tmp_path):
    config = small_config()
    session = DQNSession(config)
    with pytest.raises(ValueError, match="validation"):
        session.report()
    restored = load_dqn(save_dqn(session, ArtifactStore(tmp_path)), config)
    assert restored.returns == []
    restored.advance(1)
    assert len(restored.returns) == 1
    with pytest.raises(ValueError):
        restored.advance(1)
    with pytest.raises(ValueError):
        DQNSession(ControlConfig(method="ppo"))


@pytest.fixture
def checkpoint(tmp_path):
    config = small_config()
    session = DQNSession(config)
    session.advance(6)
    path = save_dqn(session, ArtifactStore(tmp_path))
    return path, config, session


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "checksum",
        "unknown",
        "shape",
        "names",
        "nan",
        "optimizer_count",
        "optimizer_shape",
        "optimizer_step",
        "negative_variance",
        "replay_length",
        "replay_position",
        "replay_observation",
        "episodes",
        "steps",
        "updates",
        "selection",
        "validation",
        "score",
        "returns",
        "config",
        "runtime",
    ],
)
def test_corrupt_checkpoints_fail_without_mutating_existing_session(checkpoint, fault):
    path, config, session = checkpoint
    before = {key: tensor.clone() for key, tensor in session.model.state_dict().items()}
    data = json.loads(path.read_text())
    payload = data["payload"]
    name = next(iter(payload["model"]))
    if fault == "schema":
        data["schema_version"] = "bad/2"
    elif fault == "checksum":
        data["sha256"] = "0" * 64
    elif fault == "unknown":
        payload["execute"] = "forbidden"
    elif fault == "shape":
        payload["model"][name]["shape"] = [len(payload["model"][name]["values"])]
    elif fault == "names":
        payload["model"]["unknown"] = payload["model"].pop(name)
    elif fault == "nan":
        payload["model"][name]["values"][0] = float("nan")
    elif fault == "optimizer_count":
        payload["optimizer"].pop()
    elif fault == "optimizer_shape":
        record = payload["optimizer"][0]["exp_avg"]
        record["shape"] = [len(record["values"])]
    elif fault == "optimizer_step":
        payload["optimizer"][0]["step"]["values"] = [999.0]
    elif fault == "negative_variance":
        payload["optimizer"][0]["exp_avg_sq"]["values"][0] = -1.0
    elif fault == "replay_length":
        payload["replay"]["rewards"].pop()
    elif fault == "replay_position":
        payload["replay"]["position"] = 99
    elif fault == "replay_observation":
        payload["replay"]["observations"][0] = [1.0, 2.0]
    elif fault == "episodes":
        payload["diagnostics"].pop()
    elif fault == "steps":
        payload["environment_steps"] = 0
    elif fault == "updates":
        payload["updates"] = 999
    elif fault == "selection":
        payload["selected_episode"] = 999
    elif fault == "validation":
        payload["validation"].pop()
    elif fault == "score":
        payload["best_score"] = -999.0
    elif fault == "returns":
        payload["returns"][0] = 1.5
    elif fault == "runtime":
        payload["runtime"]["python"] = "0.0.0"
    else:
        payload["config"]["seed"] += 1
    if fault != "checksum":
        data["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    path.write_text(json.dumps(data))
    with pytest.raises(ReinforcementError, match="corrupt") as error:
        load_dqn(path, config)
    assert error.value.__cause__ is not None
    for key, tensor in before.items():
        assert torch.equal(tensor, session.model.state_dict()[key])


def test_missing_truncated_oversized_and_nonfinite_tensor(tmp_path, monkeypatch):
    config = small_config()
    with pytest.raises(ReinforcementError):
        load_dqn(tmp_path / "missing", config)
    path = tmp_path / "broken.json"
    path.write_bytes(b'{"unfinished":')
    with pytest.raises(ReinforcementError):
        load_dqn(path, config)
    monkeypatch.setattr("learning_atlas.reinforcement.checkpoints._MAX_BYTES", 2)
    with pytest.raises(ReinforcementError):
        load_dqn(path, config)
    with pytest.raises(ReinforcementError, match="storage budget"):
        save_dqn(DQNSession(config), ArtifactStore(tmp_path), "large.json")
    with pytest.raises(ReinforcementError, match="float32"):
        TensorRecord.capture(torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValidationError):
        TensorRecord(shape=[2], values=[1.0])
    with pytest.raises(ValidationError):
        TensorRecord(shape=[1], values=[1e100])


@pytest.mark.parametrize("replay,target", [(False, True), (True, False), (False, False)])
def test_dqn_ablations_execute_same_update_schedule(replay, target):
    config = small_config(replay_enabled=replay, target_enabled=target)
    session = DQNSession(config)
    session.advance(config.episodes)
    assert session.updates > 0
    assert len(session.report().returns) == config.episodes
    if not target:
        fresh = DQNSession(config)
        for key, value in fresh.target.state_dict().items():
            assert torch.equal(value, session.target.state_dict()[key])
