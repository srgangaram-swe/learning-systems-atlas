"""Behavioral contracts for the shared deterministic PyTorch trainer."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from learning_atlas.deep.training import (
    Batch,
    ClassificationObjective,
    DeepTrainingError,
    ReconstructionObjective,
    SequenceClassificationObjective,
    Trainer,
    TrainerConfig,
    resolve_device,
    seed_torch,
    tensor_dataset,
)

pytestmark = pytest.mark.unit


def _classification_dataset() -> TensorDataset:
    features = torch.tensor(
        [
            [-1.5, -1.0],
            [-1.0, -1.5],
            [-0.8, -0.4],
            [-0.4, -0.8],
            [0.4, 0.8],
            [0.8, 0.4],
            [1.0, 1.5],
            [1.5, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return TensorDataset(features, labels)


def _linear_model(*, seed: int, outputs: int = 2) -> nn.Linear:
    previous_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        return nn.Linear(2, outputs)
    finally:
        torch.set_rng_state(previous_state)


def _trainer_config(*, max_epochs: int = 4, **overrides: object) -> TrainerConfig:
    values: dict[str, object] = {
        "max_epochs": max_epochs,
        "batch_size": 2,
        "learning_rate": 0.05,
        "seed": 991,
        "optimizer": "sgd",
        "momentum": 0.25,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainerConfig(**values)  # type: ignore[arg-type]


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _assert_state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0.0, atol=0.0)


class _PlateauObjective:
    """A finite scalar objective whose validation loss never improves."""

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        inputs, _ = batch
        return model(inputs).sum() * 0.0 + 1.0

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model
        examples = sum(int(batch[0].shape[0]) for batch in batches)
        return {"examples": float(examples)}


class _NonScalarObjective:
    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        inputs, _ = batch
        return model(inputs)

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model, batches
        return {}


class _NonFiniteLossObjective:
    def __init__(self, value: float) -> None:
        self._value = value

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        inputs, _ = batch
        return model(inputs).sum() * 0.0 + self._value

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model, batches
        return {}


class _NonFiniteMetricObjective:
    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        inputs, labels = batch
        return nn.functional.cross_entropy(model(inputs), labels)

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model, batches
        return {"bad_metric": math.nan}


class _InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.detach() * 0.0

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (torch.full_like(gradient, math.inf),)


class _NonFiniteGradientObjective:
    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        inputs, _ = batch
        return _InfiniteGradient.apply(model(inputs).sum())

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model, batches
        return {}


class _DetachedLossObjective:
    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        del model
        return batch[0].new_tensor(0.0)

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model, batches
        return {}


class _SquaredErrorObjective:
    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        inputs, targets = batch
        return torch.mean(torch.square(model(inputs) - targets))

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model, batches
        return {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_epochs", 0),
        ("max_epochs", True),
        ("max_epochs", 1.5),
        ("batch_size", 0),
        ("batch_size", False),
        ("batch_size", 2.5),
        ("learning_rate", 0.0),
        ("learning_rate", math.nan),
        ("learning_rate", math.inf),
        ("learning_rate", True),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", True),
        ("seed", 1.5),
        ("optimizer", "rmsprop"),
        ("momentum", -0.1),
        ("momentum", 1.0),
        ("momentum", math.nan),
        ("momentum", True),
        ("weight_decay", -0.1),
        ("weight_decay", math.inf),
        ("weight_decay", False),
        ("patience", 0),
        ("patience", True),
        ("patience", 1.5),
        ("min_delta", -0.1),
        ("min_delta", math.nan),
        ("min_delta", False),
        ("grad_clip_norm", 0.0),
        ("grad_clip_norm", math.inf),
        ("grad_clip_norm", True),
        ("device", "gpu"),
    ],
)
def test_trainer_config_rejects_invalid_runtime_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "max_epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.05,
        "seed": 7,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        TrainerConfig(**values)  # type: ignore[arg-type]


def test_trainer_config_accepts_both_supported_optimizers() -> None:
    assert _trainer_config(optimizer="adam").optimizer == "adam"
    assert _trainer_config(optimizer="sgd").optimizer == "sgd"


def test_seed_torch_is_repeatable_and_validates_seed() -> None:
    previous_state = torch.get_rng_state()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    try:
        seed_torch(123)
        first = torch.rand(5)
        seed_torch(123)
        second = torch.rand(5)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.set_rng_state(previous_state)
        torch.use_deterministic_algorithms(deterministic_before)

    for invalid in (True, -1, 2**32, 1.5):
        with pytest.raises(ValueError, match="seed"):
            seed_torch(invalid)  # type: ignore[arg-type]


def test_cpu_device_is_explicit_and_unknown_device_is_rejected() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    trainer = Trainer(_trainer_config(device="cpu"))
    assert trainer.device == torch.device("cpu")

    with pytest.raises(ValueError, match="device"):
        resolve_device("tpu")  # type: ignore[arg-type]


def test_tensor_dataset_preserves_aligned_tensors() -> None:
    features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    targets = torch.arange(6, dtype=torch.long)

    dataset = tensor_dataset(features, targets)

    assert len(dataset) == 6
    observed_features, observed_target = dataset[4]
    torch.testing.assert_close(observed_features, features[4])
    torch.testing.assert_close(observed_target, targets[4])


def test_tensor_dataset_rejects_missing_misaligned_scalar_empty_and_nonfinite_data() -> None:
    with pytest.raises(ValueError, match="at least one"):
        tensor_dataset()
    with pytest.raises(ValueError, match="sample count"):
        tensor_dataset(torch.ones(3, 2), torch.ones(2))
    with pytest.raises(ValueError, match=r"dimension|scalar"):
        tensor_dataset(torch.tensor(1.0))
    with pytest.raises(ValueError, match=r"empty|non-empty|at least one sample"):
        tensor_dataset(torch.empty(0, 2))
    with pytest.raises(ValueError, match="finite"):
        tensor_dataset(torch.tensor([[math.nan]], dtype=torch.float32))
    with pytest.raises(TypeError, match="tensor"):
        tensor_dataset("not-a-tensor")  # type: ignore[arg-type]


def test_fit_is_exactly_repeatable_on_cpu() -> None:
    dataset = _classification_dataset()
    objective = ClassificationObjective()
    first = _linear_model(seed=71)
    second = _linear_model(seed=71)

    first_report = Trainer(_trainer_config()).fit(first, dataset, dataset, objective)
    second_report = Trainer(_trainer_config()).fit(second, dataset, dataset, objective)

    assert first_report == second_report
    _assert_state_equal(_state(first), _state(second))
    assert all(parameter.device.type == "cpu" for parameter in first.parameters())


@pytest.mark.parametrize(("enabled", "warn_only"), [(False, False), (True, True)])
def test_fit_restores_process_global_rng_and_determinism_state(
    enabled: bool,
    warn_only: bool,
) -> None:
    dataset = _classification_dataset()
    previous_rng = torch.get_rng_state()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.manual_seed(8_181)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        expected_rng = torch.get_rng_state().clone()

        Trainer(_trainer_config(max_epochs=1)).fit(
            _linear_model(seed=2), dataset, dataset, ClassificationObjective()
        )

        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0.0, atol=0.0)
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    finally:
        torch.set_rng_state(previous_rng)
        torch.use_deterministic_algorithms(
            previous_determinism,
            warn_only=previous_warn_only,
        )


def test_early_stopping_obeys_patience_and_restores_best_weights() -> None:
    dataset = _classification_dataset()
    model = _linear_model(seed=9)
    initial = _state(model)
    trainer = Trainer(_trainer_config(max_epochs=20, patience=2, min_delta=0.0))

    report = trainer.fit(model, dataset, dataset, _PlateauObjective())

    assert report.stopped_early
    assert len(report.history) == 3
    assert report.best_epoch == 1
    assert report.best_validation_loss == pytest.approx(1.0)
    assert [record.epoch for record in report.history] == [1, 2, 3]
    _assert_state_equal(initial, _state(model))


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_resume_matches_uninterrupted_training_exactly(tmp_path: Path, optimizer: str) -> None:
    dataset = _classification_dataset()
    objective = ClassificationObjective()

    uninterrupted = _linear_model(seed=31)
    uninterrupted_report = Trainer(_trainer_config(max_epochs=4, optimizer=optimizer)).fit(
        uninterrupted, dataset, dataset, objective
    )

    checkpoint = tmp_path / "trainer.pt"
    partial = _linear_model(seed=31)
    Trainer(_trainer_config(max_epochs=2, optimizer=optimizer)).fit(
        partial,
        dataset,
        dataset,
        objective,
        checkpoint_path=checkpoint,
    )
    resumed = _linear_model(seed=999)
    resumed_report = Trainer(_trainer_config(max_epochs=4, optimizer=optimizer)).fit(
        resumed,
        dataset,
        dataset,
        objective,
        resume_from=checkpoint,
    )

    assert resumed_report.resumed_from_epoch == 2
    assert resumed_report.history == uninterrupted_report.history
    assert resumed_report.best_epoch == uninterrupted_report.best_epoch
    assert resumed_report.best_validation_loss == uninterrupted_report.best_validation_loss
    _assert_state_equal(_state(resumed), _state(uninterrupted))


def test_equivalent_runs_publish_byte_identical_checkpoints(tmp_path: Path) -> None:
    dataset = _classification_dataset()
    objective = ClassificationObjective()
    paths = (tmp_path / "first.pt", tmp_path / "second.pt")

    for path in paths:
        Trainer(_trainer_config(max_epochs=2)).fit(
            _linear_model(seed=37),
            dataset,
            dataset,
            objective,
            checkpoint_path=path,
        )

    assert (
        hashlib.sha256(paths[0].read_bytes()).digest()
        == hashlib.sha256(paths[1].read_bytes()).digest()
    )


def test_checkpoint_publication_failure_preserves_existing_file_and_cleans_temporary_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "trainer.pt"
    original = b"previous-known-good-checkpoint"
    checkpoint.write_bytes(original)

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replace failure"):
        Trainer(_trainer_config(max_epochs=1)).fit(
            _linear_model(seed=38),
            _classification_dataset(),
            _classification_dataset(),
            ClassificationObjective(),
            checkpoint_path=checkpoint,
        )

    assert checkpoint.read_bytes() == original
    assert not tuple(tmp_path.glob(f".{checkpoint.name}.*.tmp"))


def test_checkpoint_loader_rejects_missing_corrupt_unsupported_and_incomplete_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(DeepTrainingError, match="does not exist"):
        Trainer.load_checkpoint(missing)

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(DeepTrainingError, match="unreadable"):
        Trainer.load_checkpoint(corrupt)

    unsupported = tmp_path / "unsupported.pt"
    torch.save({"schema": "some-other-schema"}, unsupported)
    with pytest.raises(DeepTrainingError, match="unsupported"):
        Trainer.load_checkpoint(unsupported)

    valid = tmp_path / "valid.pt"
    dataset = _classification_dataset()
    Trainer(_trainer_config(max_epochs=1)).fit(
        _linear_model(seed=3),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    raw_payload = torch.load(valid, map_location="cpu", weights_only=True)
    assert isinstance(raw_payload, dict)
    raw_payload.pop("history")
    incomplete = tmp_path / "incomplete.pt"
    torch.save(raw_payload, incomplete)
    with pytest.raises(DeepTrainingError, match="missing"):
        Trainer.load_checkpoint(incomplete)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", "two"),
        ("best_validation_loss", math.nan),
        ("epochs_without_improvement", -1),
    ],
)
def test_checkpoint_loader_validates_field_types_and_invariants(
    tmp_path: Path, field: str, value: object
) -> None:
    dataset = _classification_dataset()
    valid = tmp_path / "valid.pt"
    Trainer(_trainer_config(max_epochs=1)).fit(
        _linear_model(seed=4),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    payload: dict[str, Any] = dict(Trainer.load_checkpoint(valid))
    payload[field] = value
    malformed = tmp_path / f"bad-{field}.pt"
    torch.save(payload, malformed)

    with pytest.raises(DeepTrainingError, match=field):
        Trainer.load_checkpoint(malformed)


def test_checkpoint_loader_rejects_unexpected_fields(tmp_path: Path) -> None:
    dataset = _classification_dataset()
    valid = tmp_path / "valid.pt"
    Trainer(_trainer_config(max_epochs=1)).fit(
        _linear_model(seed=5),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    payload: dict[str, Any] = dict(Trainer.load_checkpoint(valid))
    payload["unexpected"] = "field"
    malformed = tmp_path / "unexpected.pt"
    torch.save(payload, malformed)

    with pytest.raises(DeepTrainingError, match="unexpected"):
        Trainer.load_checkpoint(malformed)


@pytest.mark.parametrize("signature", ["trainer_signature", "model_signature"])
def test_checkpoint_loader_rejects_ambiguous_signature_tensors_before_model_mutation(
    tmp_path: Path,
    signature: str,
) -> None:
    dataset = _classification_dataset()
    valid = tmp_path / "valid.pt"
    Trainer(_trainer_config(max_epochs=1)).fit(
        _linear_model(seed=51),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    if signature == "trainer_signature":
        trainer_signature = payload[signature]
        assert isinstance(trainer_signature, dict)
        trainer_signature["batch_size"] = torch.tensor([1, 2])
    else:
        model_signature = payload[signature]
        assert isinstance(model_signature, dict)
        first_name = next(iter(model_signature))
        model_signature[first_name] = (torch.tensor([1, 2]), "torch.float32")
    malformed = tmp_path / f"bad-{signature}.pt"
    torch.save(payload, malformed)

    with pytest.raises(DeepTrainingError, match=signature):
        Trainer.load_checkpoint(malformed)

    model = _linear_model(seed=151)
    before = _state(model)
    with pytest.raises(DeepTrainingError, match=signature):
        Trainer(_trainer_config(max_epochs=2)).fit(
            model,
            dataset,
            dataset,
            ClassificationObjective(),
            resume_from=malformed,
        )
    _assert_state_equal(before, _state(model))


@pytest.mark.parametrize("corruption", ["rng", "best_shape", "optimizer_nan"])
def test_checkpoint_loader_rejects_deeply_malformed_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    dataset = _classification_dataset()
    valid = tmp_path / "valid.pt"
    Trainer(_trainer_config(max_epochs=1, optimizer="adam")).fit(
        _linear_model(seed=6),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    if corruption == "rng":
        payload["torch_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    elif corruption == "best_shape":
        best_state = payload["best_model_state"]
        assert isinstance(best_state, dict)
        first_name = next(iter(best_state))
        best_state[first_name] = torch.zeros(1)
    else:
        optimizer_state = payload["optimizer_state"]
        assert isinstance(optimizer_state, dict)
        states = optimizer_state["state"]
        assert isinstance(states, dict) and states
        parameter_state = next(iter(states.values()))
        assert isinstance(parameter_state, dict)
        tensor = next(
            value for value in parameter_state.values() if isinstance(value, torch.Tensor)
        )
        tensor.fill_(math.nan)
    malformed = tmp_path / f"{corruption}.pt"
    torch.save(payload, malformed)

    with pytest.raises(DeepTrainingError, match=r"[Rr][Nn][Gg]|shape|optimizer_state|non-finite"):
        Trainer.load_checkpoint(malformed)


def test_resume_rejects_optimizer_incompatibility_before_model_mutation(tmp_path: Path) -> None:
    dataset = _classification_dataset()
    valid = tmp_path / "valid.pt"
    Trainer(_trainer_config(max_epochs=1, optimizer="adam")).fit(
        _linear_model(seed=7),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    optimizer_state = payload["optimizer_state"]
    assert isinstance(optimizer_state, dict)
    groups = optimizer_state["param_groups"]
    assert isinstance(groups, list)
    params = groups[0]["params"]
    assert isinstance(params, list)
    params.append(999)
    malformed = tmp_path / "optimizer-incompatible.pt"
    torch.save(payload, malformed)

    model = _linear_model(seed=99)
    before = _state(model)
    with pytest.raises(DeepTrainingError, match=r"checkpoint.*state"):
        Trainer(_trainer_config(max_epochs=2, optimizer="adam")).fit(
            model,
            dataset,
            dataset,
            ClassificationObjective(),
            resume_from=malformed,
        )
    _assert_state_equal(before, _state(model))


def test_resume_rejects_finite_wrong_shaped_optimizer_slots_before_model_mutation(
    tmp_path: Path,
) -> None:
    dataset = _classification_dataset()
    valid = tmp_path / "valid.pt"
    Trainer(_trainer_config(max_epochs=1, optimizer="adam")).fit(
        _linear_model(seed=8),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    optimizer_state = payload["optimizer_state"]
    assert isinstance(optimizer_state, dict)
    states = optimizer_state["state"]
    assert isinstance(states, dict)
    second_parameter = list(states)[1]
    parameter_slots = states[second_parameter]
    assert isinstance(parameter_slots, dict)
    parameter_slots["exp_avg"] = torch.zeros(1)
    malformed = tmp_path / "wrong-shaped-slot.pt"
    torch.save(payload, malformed)

    model = _linear_model(seed=108)
    before = _state(model)
    with pytest.raises(DeepTrainingError, match=r"optimizer state.*shape"):
        Trainer(_trainer_config(max_epochs=2, optimizer="adam")).fit(
            model,
            dataset,
            dataset,
            ClassificationObjective(),
            resume_from=malformed,
        )

    _assert_state_equal(before, _state(model))


@pytest.mark.parametrize(
    "corruption",
    [
        "group_option_type",
        "group_tuple_arity",
        "extra_group",
        "group_schema",
        "adam_slot_schema",
        "adam_step_rank",
        "adam_step_dtype",
        "adam_step_negative",
        "adam_slot_type",
        "trainer_schema",
        "trainer_optimizer",
        "model_entry",
        "model_shape",
    ],
)
def test_resume_rejects_adversarial_exact_schema_corruption_before_model_mutation(
    tmp_path: Path,
    corruption: str,
) -> None:
    dataset = _classification_dataset()
    valid = tmp_path / "valid.pt"
    Trainer(_trainer_config(max_epochs=1, optimizer="adam")).fit(
        _linear_model(seed=9),
        dataset,
        dataset,
        ClassificationObjective(),
        checkpoint_path=valid,
    )
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    optimizer_state = payload["optimizer_state"]
    assert isinstance(optimizer_state, dict)
    groups = optimizer_state["param_groups"]
    states = optimizer_state["state"]
    assert isinstance(groups, list) and groups
    assert isinstance(states, dict) and states
    group = groups[0]
    slots = next(iter(states.values()))
    assert isinstance(group, dict)
    assert isinstance(slots, dict)
    if corruption == "group_option_type":
        group["lr"] = "0.05"
    elif corruption == "group_tuple_arity":
        group["betas"] = (0.9,)
    elif corruption == "extra_group":
        groups.append(dict(group))
    elif corruption == "group_schema":
        group.pop("eps")
    elif corruption == "adam_slot_schema":
        slots.pop("exp_avg_sq")
    elif corruption == "adam_step_rank":
        slots["step"] = torch.ones(1)
    elif corruption == "adam_step_dtype":
        slots["step"] = torch.tensor(1, dtype=torch.int64)
    elif corruption == "adam_step_negative":
        slots["step"] = torch.tensor(-1.0)
    elif corruption == "adam_slot_type":
        slots["exp_avg"] = "not-a-tensor"
    elif corruption == "trainer_schema":
        trainer_signature = payload["trainer_signature"]
        assert isinstance(trainer_signature, dict)
        trainer_signature.pop("batch_size")
    elif corruption == "trainer_optimizer":
        trainer_signature = payload["trainer_signature"]
        assert isinstance(trainer_signature, dict)
        trainer_signature["optimizer"] = "rmsprop"
    elif corruption == "model_entry":
        model_signature = payload["model_signature"]
        assert isinstance(model_signature, dict)
        first_name = next(iter(model_signature))
        model_signature[first_name] = list(model_signature[first_name])
    else:
        model_signature = payload["model_signature"]
        assert isinstance(model_signature, dict)
        first_name = next(iter(model_signature))
        _, dtype = model_signature[first_name]
        model_signature[first_name] = ((999,), dtype)
    malformed = tmp_path / f"{corruption}.pt"
    torch.save(payload, malformed)

    model = _linear_model(seed=109)
    before = _state(model)
    with pytest.raises(DeepTrainingError, match="checkpoint"):
        Trainer(_trainer_config(max_epochs=2, optimizer="adam")).fit(
            model,
            dataset,
            dataset,
            ClassificationObjective(),
            resume_from=malformed,
        )

    _assert_state_equal(before, _state(model))


def test_resume_rejects_terminal_early_stopping_checkpoint_without_model_mutation(
    tmp_path: Path,
) -> None:
    dataset = _classification_dataset()
    checkpoint = tmp_path / "early-stopped.pt"
    report = Trainer(_trainer_config(max_epochs=20, patience=2)).fit(
        _linear_model(seed=12),
        dataset,
        dataset,
        _PlateauObjective(),
        checkpoint_path=checkpoint,
    )
    assert report.stopped_early
    assert len(report.history) == 3

    model = _linear_model(seed=112)
    before = _state(model)
    with pytest.raises(DeepTrainingError, match="terminal early-stopping state"):
        Trainer(_trainer_config(max_epochs=30, patience=2)).fit(
            model,
            dataset,
            dataset,
            _PlateauObjective(),
            resume_from=checkpoint,
        )

    _assert_state_equal(before, _state(model))


def test_resume_rejects_incompatible_model_optimizer_and_completed_checkpoint(
    tmp_path: Path,
) -> None:
    dataset = _classification_dataset()
    objective = ClassificationObjective()
    checkpoint = tmp_path / "trainer.pt"
    Trainer(_trainer_config(max_epochs=2, optimizer="sgd")).fit(
        _linear_model(seed=11),
        dataset,
        dataset,
        objective,
        checkpoint_path=checkpoint,
    )

    with pytest.raises(DeepTrainingError, match=r"model|checkpoint"):
        Trainer(_trainer_config(max_epochs=3, optimizer="sgd")).fit(
            _linear_model(seed=11, outputs=3),
            dataset,
            dataset,
            objective,
            resume_from=checkpoint,
        )
    with pytest.raises(DeepTrainingError, match=r"optimizer|checkpoint"):
        Trainer(_trainer_config(max_epochs=3, optimizer="adam")).fit(
            _linear_model(seed=11),
            dataset,
            dataset,
            objective,
            resume_from=checkpoint,
        )
    with pytest.raises(DeepTrainingError, match="already covers"):
        Trainer(_trainer_config(max_epochs=2, optimizer="sgd")).fit(
            _linear_model(seed=11),
            dataset,
            dataset,
            objective,
            resume_from=checkpoint,
        )


def test_empty_training_and_evaluation_datasets_fail_with_domain_errors() -> None:
    empty = TensorDataset(
        torch.empty((0, 2), dtype=torch.float32), torch.empty(0, dtype=torch.long)
    )
    dataset = _classification_dataset()
    trainer = Trainer(_trainer_config(max_epochs=1))

    with pytest.raises(DeepTrainingError, match="training dataset"):
        trainer.fit(_linear_model(seed=1), empty, dataset, ClassificationObjective())
    with pytest.raises(DeepTrainingError, match="evaluation dataset"):
        trainer.evaluate(_linear_model(seed=1), empty, ClassificationObjective())


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 2]),
        torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]]),
    ],
)
def test_classification_label_contracts_raise_domain_errors_without_updates(
    labels: torch.Tensor,
) -> None:
    features, _ = _classification_dataset().tensors
    dataset = TensorDataset(features, labels)
    model = _linear_model(seed=101)
    before = _state(model)

    with pytest.raises(DeepTrainingError, match=r"labels.*(int64|shape|dimensional|lie)"):
        Trainer(_trainer_config(max_epochs=1)).fit(
            model,
            dataset,
            dataset,
            ClassificationObjective(),
        )

    _assert_state_equal(before, _state(model))


class _InvalidLogitModel(nn.Module):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self._failure = failure

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._failure == "rank":
            return torch.zeros(inputs.shape[0]) + self.anchor
        if self._failure == "dtype":
            return torch.zeros((inputs.shape[0], 2), dtype=torch.int64)
        return torch.full((inputs.shape[0], 2), math.nan) + self.anchor


@pytest.mark.parametrize("failure", ["rank", "dtype", "nonfinite"])
def test_classification_logit_contracts_raise_domain_errors(failure: str) -> None:
    dataset = _classification_dataset()

    with pytest.raises(DeepTrainingError, match=r"logits.*(shape|floating|non-finite)"):
        Trainer(_trainer_config(max_epochs=1)).fit(
            _InvalidLogitModel(failure),
            dataset,
            dataset,
            ClassificationObjective(),
        )


def test_reconstruction_objective_rejects_shape_mismatch_actionably() -> None:
    dataset = TensorDataset(torch.ones(4, 2))

    with pytest.raises(DeepTrainingError, match="exactly match"):
        Trainer(_trainer_config(max_epochs=1)).fit(
            nn.Linear(2, 1),
            dataset,
            dataset,
            ReconstructionObjective(),
        )


class _NonFiniteBufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("audit_buffer", torch.tensor(math.nan))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_trainer_rejects_nonfinite_persistent_buffers_before_training() -> None:
    with pytest.raises(DeepTrainingError, match=r"model state.*audit_buffer.*non-finite"):
        Trainer(_trainer_config(max_epochs=1)).fit(
            _NonFiniteBufferModel(),
            _classification_dataset(),
            _classification_dataset(),
            ClassificationObjective(),
        )


@pytest.mark.parametrize(
    ("objective", "batch", "context"),
    [
        (ClassificationObjective(), (torch.ones(2, 2),), "classification"),
        (
            SequenceClassificationObjective(),
            (torch.ones(2, 3, 1), torch.ones(2, dtype=torch.long)),
            "sequence",
        ),
        (ReconstructionObjective(), (torch.ones(2, 2), torch.ones(2, 2)), "reconstruction"),
    ],
)
def test_objectives_reject_wrong_batch_arity(objective: object, batch: Batch, context: str) -> None:
    with pytest.raises(DeepTrainingError, match=context):
        objective.loss(nn.Identity(), batch)  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_training_loss_stops_before_model_update(value: float) -> None:
    dataset = _classification_dataset()
    model = _linear_model(seed=81)
    initial = _state(model)

    with pytest.raises(DeepTrainingError, match=r"training loss.*non-finite"):
        Trainer(_trainer_config(max_epochs=1)).fit(
            model, dataset, dataset, _NonFiniteLossObjective(value)
        )

    _assert_state_equal(initial, _state(model))


def test_non_scalar_and_detached_losses_raise_actionable_domain_errors() -> None:
    dataset = _classification_dataset()
    trainer = Trainer(_trainer_config(max_epochs=1))

    with pytest.raises(DeepTrainingError, match="scalar"):
        trainer.fit(_linear_model(seed=2), dataset, dataset, _NonScalarObjective())
    with pytest.raises(DeepTrainingError, match=r"gradient|requires"):
        trainer.fit(_linear_model(seed=2), dataset, dataset, _DetachedLossObjective())


def test_nonfinite_gradient_is_detected_before_parameters_are_corrupted() -> None:
    dataset = _classification_dataset()
    model = _linear_model(seed=91)

    with pytest.raises(DeepTrainingError, match=r"gradient.*non-finite|non-finite.*gradient"):
        Trainer(_trainer_config(max_epochs=1)).fit(
            model, dataset, dataset, _NonFiniteGradientObjective()
        )

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_nonfinite_evaluation_metric_is_rejected() -> None:
    dataset = _classification_dataset()

    with pytest.raises(DeepTrainingError, match=r"bad_metric.*non-finite"):
        Trainer(_trainer_config(max_epochs=1)).evaluate(
            _linear_model(seed=3), dataset, _NonFiniteMetricObjective()
        )


def test_evaluation_uses_sample_weighted_batch_means() -> None:
    inputs = torch.zeros((5, 1), dtype=torch.float32)
    targets = torch.tensor([[1.0], [1.0], [1.0], [9.0], [9.0]])
    dataset = TensorDataset(inputs, targets)
    trainer = Trainer(_trainer_config(batch_size=4))

    loss, metrics = trainer.evaluate(nn.Identity(), dataset, _SquaredErrorObjective())

    assert loss == pytest.approx((1.0 + 1.0 + 1.0 + 81.0 + 81.0) / 5.0)
    assert metrics == {}
