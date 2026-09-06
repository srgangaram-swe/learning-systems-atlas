"""Deterministic, checkpointable training infrastructure shared by PyTorch models.

The :class:`Trainer` owns the optimizer, the shuffling RNG, checkpoint
persistence, early stopping, and device placement, so every PyTorch study in
the atlas trains through one audited loop instead of re-implementing it. All
random state (global torch RNG plus the loader-shuffle generator) is captured
in each checkpoint, which makes resume-from-checkpoint bit-reproducible on the
same platform.
"""

from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections.abc import Iterator, Mapping, Sized
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

Batch = tuple[torch.Tensor, ...]
_CHECKPOINT_SCHEMA = "learning-atlas-trainer/2"


class CheckpointPayload(TypedDict):
    """On-disk trainer state captured after a completed epoch."""

    schema: str
    epoch: int
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, object]
    torch_rng_state: torch.Tensor
    shuffle_rng_state: torch.Tensor
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    best_model_state: dict[str, torch.Tensor]
    epochs_without_improvement: int
    trainer_signature: dict[str, object]
    model_signature: dict[str, tuple[tuple[int, ...], str]]


class DeepTrainingError(RuntimeError):
    """Raised when a training run violates its numerical or state contracts."""


class BatchObjective(Protocol):
    """Task-specific differentiable loss at the shared numerical-step boundary."""

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Return the mean minibatch loss."""


class Objective(BatchObjective, Protocol):
    """Dataset objective with auxiliary evaluation metrics."""

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        """Return auxiliary metrics for an evaluation pass."""


def seed_torch(seed: int) -> None:
    """Seed the global torch RNG and require deterministic kernel selection."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        msg = "seed must be an integer in [0, 2**32 - 1]"
        raise ValueError(msg)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: Literal["cpu", "auto"]) -> torch.device:
    """Resolve the execution device; reference profiles pin ``cpu``."""

    if requested not in {"cpu", "auto"}:
        msg = f"device must be 'cpu' or 'auto'; received {requested!r}"
        raise ValueError(msg)
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Validated hyperparameters for one deterministic training run."""

    max_epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    optimizer: Literal["adam", "sgd"] = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    patience: int | None = None
    min_delta: float = 0.0
    grad_clip_norm: float | None = None
    device: Literal["cpu", "auto"] = "cpu"

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_epochs, bool)
            or not isinstance(self.max_epochs, int)
            or self.max_epochs < 1
        ):
            msg = "max_epochs must be a positive integer"
            raise ValueError(msg)
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            msg = "batch_size must be a positive integer"
            raise ValueError(msg)
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            msg = "learning_rate must be a finite positive value"
            raise ValueError(msg)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            msg = "seed must fit in an unsigned 32-bit integer"
            raise ValueError(msg)
        if self.optimizer not in {"adam", "sgd"}:
            msg = "optimizer must be 'adam' or 'sgd'"
            raise ValueError(msg)
        if (
            isinstance(self.momentum, bool)
            or not isinstance(self.momentum, (int, float))
            or not math.isfinite(self.momentum)
            or not 0.0 <= self.momentum < 1.0
        ):
            msg = "momentum must lie in [0, 1)"
            raise ValueError(msg)
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            msg = "weight_decay must be a finite non-negative value"
            raise ValueError(msg)
        if self.patience is not None and (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience < 1
        ):
            msg = "patience must be a positive integer when provided"
            raise ValueError(msg)
        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, (int, float))
            or not math.isfinite(self.min_delta)
            or self.min_delta < 0.0
        ):
            msg = "min_delta must be a finite non-negative value"
            raise ValueError(msg)
        if self.grad_clip_norm is not None and (
            isinstance(self.grad_clip_norm, bool)
            or not isinstance(self.grad_clip_norm, (int, float))
            or not math.isfinite(self.grad_clip_norm)
            or self.grad_clip_norm <= 0.0
        ):
            msg = "grad_clip_norm must be a finite positive value when provided"
            raise ValueError(msg)
        if self.device not in {"cpu", "auto"}:
            msg = "device must be 'cpu' or 'auto'"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Loss and metric evidence for one completed epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Immutable summary of a completed (possibly early-stopped) training run."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    resumed_from_epoch: int | None


def _checkpoint_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"checkpoint field {field!r} must be an integer >= {minimum}"
        raise DeepTrainingError(msg)
    return value


def _checkpoint_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"checkpoint field {field!r} must be finite"
        raise DeepTrainingError(msg)
    numeric = float(value)
    if not math.isfinite(numeric):
        msg = f"checkpoint field {field!r} must be finite"
        raise DeepTrainingError(msg)
    return numeric


def _checkpoint_state(value: object, *, field: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        msg = f"checkpoint field {field!r} must be a non-empty tensor mapping"
        raise DeepTrainingError(msg)
    validated: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            msg = f"checkpoint field {field!r} contains an invalid state entry"
            raise DeepTrainingError(msg)
        if not bool(torch.all(torch.isfinite(tensor))):
            msg = f"checkpoint field {field!r} contains non-finite tensor {name!r}"
            raise DeepTrainingError(msg)
        validated[name] = tensor
    return validated


def _validate_checkpoint_tree(value: object, *, field: str, depth: int = 0) -> None:
    """Reject unsupported or non-finite values in optimizer state."""

    if depth > 32:
        msg = f"checkpoint field {field!r} is nested too deeply"
        raise DeepTrainingError(msg)
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"checkpoint field {field!r} contains a non-finite value"
            raise DeepTrainingError(msg)
        return
    if isinstance(value, torch.Tensor):
        if not bool(torch.all(torch.isfinite(value))):
            msg = f"checkpoint field {field!r} contains a non-finite tensor"
            raise DeepTrainingError(msg)
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                msg = f"checkpoint field {field!r} contains an unsupported mapping key"
                raise DeepTrainingError(msg)
            _validate_checkpoint_tree(nested, field=f"{field}.{key}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_checkpoint_tree(nested, field=f"{field}[{index}]", depth=depth + 1)
        return
    msg = f"checkpoint field {field!r} contains unsupported state type {type(value).__name__}"
    raise DeepTrainingError(msg)


def _checkpoint_optimizer_state(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"state", "param_groups"}:
        msg = "checkpoint optimizer_state must contain exactly state and param_groups"
        raise DeepTrainingError(msg)
    if not isinstance(value["state"], dict):
        msg = "checkpoint optimizer_state.state must be a mapping"
        raise DeepTrainingError(msg)
    if not isinstance(value["param_groups"], list) or not value["param_groups"]:
        msg = "checkpoint optimizer_state.param_groups must be a non-empty list"
        raise DeepTrainingError(msg)
    _validate_checkpoint_tree(value, field="optimizer_state")
    return cast(dict[str, object], value)


def _optimizer_state_error(detail: str) -> DeepTrainingError:
    return DeepTrainingError(f"checkpoint optimizer state is incompatible: {detail}")


def _same_optimizer_option(observed: object, expected: object) -> bool:
    """Compare scalar/container optimizer options without tensor truth coercion."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, tuple):
        if not isinstance(observed, tuple) or len(observed) != len(expected):
            return False
        return all(
            _same_optimizer_option(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return bool(observed == expected)


def _optimizer_slot_tensor(
    value: object,
    parameter: torch.Tensor,
    *,
    field: str,
    scalar: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _optimizer_state_error(f"slot {field!r} must be a tensor")
    if scalar:
        if value.ndim != 0:
            raise _optimizer_state_error(f"slot {field!r} must be scalar")
        return value
    if value.shape != parameter.shape:
        raise _optimizer_state_error(
            f"slot {field!r} shape {tuple(value.shape)} does not match "
            f"parameter shape {tuple(parameter.shape)}"
        )
    if value.dtype != parameter.dtype:
        raise _optimizer_state_error(
            f"slot {field!r} dtype {value.dtype} does not match parameter dtype {parameter.dtype}"
        )
    if value.device != parameter.device:
        raise _optimizer_state_error(
            f"slot {field!r} device {value.device} does not match parameter device "
            f"{parameter.device}"
        )
    return value


def _load_validated_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: dict[str, object],
) -> None:
    """Validate this Trainer's exact optimizer schema before accepting resume state."""

    expected_groups = optimizer.state_dict()["param_groups"]
    observed_groups = state["param_groups"]
    if not isinstance(observed_groups, list) or len(observed_groups) != len(expected_groups):
        raise _optimizer_state_error("parameter-group count disagrees with this model")
    for index, (observed, expected) in enumerate(
        zip(observed_groups, expected_groups, strict=True)
    ):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise _optimizer_state_error(f"parameter group {index} has an unsupported schema")
        observed_parameters = observed["params"]
        expected_parameters = expected["params"]
        if (
            not isinstance(observed_parameters, list)
            or any(
                isinstance(parameter, bool) or not isinstance(parameter, int)
                for parameter in observed_parameters
            )
            or observed_parameters != expected_parameters
        ):
            raise _optimizer_state_error(
                f"parameter group {index} does not match this model's parameter order"
            )
        for option, expected_value in expected.items():
            if option == "params":
                continue
            if not _same_optimizer_option(observed[option], expected_value):
                raise _optimizer_state_error(
                    f"parameter group {index} option {option!r} disagrees with the Trainer"
                )

    try:
        optimizer.load_state_dict(state)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise _optimizer_state_error("state cannot be loaded for this model") from error

    if isinstance(optimizer, torch.optim.Adam):
        required_slots = {"step", "exp_avg", "exp_avg_sq"}
        for parameter, slots in optimizer.state.items():
            if not isinstance(slots, dict) or set(slots) != required_slots:
                raise _optimizer_state_error("Adam slots have an unsupported schema")
            step = _optimizer_slot_tensor(slots["step"], parameter, field="step", scalar=True)
            if not step.is_floating_point() or step.is_complex():
                raise _optimizer_state_error("Adam step must be a real floating-point scalar")
            if float(step.detach().cpu().item()) < 0.0:
                raise _optimizer_state_error("Adam step must be non-negative")
            _optimizer_slot_tensor(slots["exp_avg"], parameter, field="exp_avg")
            _optimizer_slot_tensor(slots["exp_avg_sq"], parameter, field="exp_avg_sq")
        return

    if not isinstance(optimizer, torch.optim.SGD):  # defensive: config currently forbids others
        raise _optimizer_state_error(f"unsupported optimizer type {type(optimizer).__name__}")
    momentum = float(optimizer.param_groups[0]["momentum"])
    if momentum == 0.0 and optimizer.state:
        raise _optimizer_state_error("momentum-free SGD must not contain parameter slots")
    for parameter, slots in optimizer.state.items():
        if not isinstance(slots, dict) or set(slots) != {"momentum_buffer"}:
            raise _optimizer_state_error("SGD slots have an unsupported schema")
        _optimizer_slot_tensor(slots["momentum_buffer"], parameter, field="momentum_buffer")


def _checkpoint_rng_state(value: object, *, field: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.numel() == 0
        or value.dtype != torch.uint8
        or value.device.type != "cpu"
    ):
        msg = f"checkpoint field {field!r} is not a valid CPU RNG state"
        raise DeepTrainingError(msg)
    try:
        probe = torch.Generator(device="cpu")
        probe.set_state(value)
    except RuntimeError as error:
        msg = f"checkpoint field {field!r} is incompatible with this Torch runtime"
        raise DeepTrainingError(msg) from error
    return value


def _checkpoint_trainer_signature(value: object) -> dict[str, object]:
    expected_fields = {
        "batch_size",
        "learning_rate",
        "seed",
        "optimizer",
        "momentum",
        "weight_decay",
        "patience",
        "min_delta",
        "grad_clip_norm",
        "device",
    }
    if (
        not isinstance(value, dict)
        or any(not isinstance(field, str) for field in value)
        or set(value) != expected_fields
    ):
        msg = "checkpoint trainer_signature has an unsupported schema"
        raise DeepTrainingError(msg)

    optimizer = value["optimizer"]
    device = value["device"]
    if not isinstance(optimizer, str) or optimizer not in {"adam", "sgd"}:
        msg = "checkpoint trainer_signature optimizer must be 'adam' or 'sgd'"
        raise DeepTrainingError(msg)
    if not isinstance(device, str) or device not in {"cpu", "auto"}:
        msg = "checkpoint trainer_signature device must be 'cpu' or 'auto'"
        raise DeepTrainingError(msg)
    raw_patience = value["patience"]
    patience = (
        None
        if raw_patience is None
        else _checkpoint_int(raw_patience, field="trainer_signature.patience", minimum=1)
    )
    raw_clip = value["grad_clip_norm"]
    grad_clip_norm = (
        None
        if raw_clip is None
        else _checkpoint_float(raw_clip, field="trainer_signature.grad_clip_norm")
    )
    seed = _checkpoint_int(value["seed"], field="trainer_signature.seed")
    if seed > 2**32 - 1:
        msg = "checkpoint trainer_signature seed must fit in an unsigned 32-bit integer"
        raise DeepTrainingError(msg)
    try:
        validated = TrainerConfig(
            max_epochs=1,
            batch_size=_checkpoint_int(
                value["batch_size"], field="trainer_signature.batch_size", minimum=1
            ),
            learning_rate=_checkpoint_float(
                value["learning_rate"], field="trainer_signature.learning_rate"
            ),
            seed=seed,
            optimizer=cast(Literal["adam", "sgd"], optimizer),
            momentum=_checkpoint_float(value["momentum"], field="trainer_signature.momentum"),
            weight_decay=_checkpoint_float(
                value["weight_decay"], field="trainer_signature.weight_decay"
            ),
            patience=patience,
            min_delta=_checkpoint_float(value["min_delta"], field="trainer_signature.min_delta"),
            grad_clip_norm=grad_clip_norm,
            device=cast(Literal["cpu", "auto"], device),
        )
    except ValueError as error:
        msg = f"checkpoint trainer_signature is invalid: {error}"
        raise DeepTrainingError(msg) from error
    return {
        "batch_size": validated.batch_size,
        "learning_rate": validated.learning_rate,
        "seed": validated.seed,
        "optimizer": validated.optimizer,
        "momentum": validated.momentum,
        "weight_decay": validated.weight_decay,
        "patience": validated.patience,
        "min_delta": validated.min_delta,
        "grad_clip_norm": validated.grad_clip_norm,
        "device": validated.device,
    }


def _checkpoint_model_signature(
    value: object,
    *,
    model_state: Mapping[str, torch.Tensor],
) -> dict[str, tuple[tuple[int, ...], str]]:
    if (
        not isinstance(value, dict)
        or any(not isinstance(name, str) for name in value)
        or set(value) != set(model_state)
    ):
        msg = "checkpoint model_signature has an unsupported schema"
        raise DeepTrainingError(msg)
    validated: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, raw_signature in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(raw_signature, tuple)
            or len(raw_signature) != 2
        ):
            msg = "checkpoint model_signature contains an invalid entry"
            raise DeepTrainingError(msg)
        raw_shape, raw_dtype = raw_signature
        if (
            not isinstance(raw_shape, tuple)
            or any(
                isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
                for extent in raw_shape
            )
            or not isinstance(raw_dtype, str)
            or not raw_dtype
        ):
            msg = f"checkpoint model_signature entry {name!r} is invalid"
            raise DeepTrainingError(msg)
        signature = (raw_shape, raw_dtype)
        expected = (tuple(model_state[name].shape), str(model_state[name].dtype))
        if signature != expected:
            msg = "checkpoint model signature does not describe its model state"
            raise DeepTrainingError(msg)
        validated[name] = signature
    return validated


def _validated_checkpoint_payload(loaded: dict[object, object], *, path: Path) -> CheckpointPayload:
    """Validate untrusted safe-loader output before exposing trainer state."""

    epoch = _checkpoint_int(loaded["epoch"], field="epoch", minimum=1)
    raw_history = loaded["history"]
    if not isinstance(raw_history, (tuple, list)) or len(raw_history) != epoch:
        msg = "checkpoint history must contain exactly one record per completed epoch"
        raise DeepTrainingError(msg)
    history: list[EpochRecord] = []
    for expected_epoch, raw_record in enumerate(raw_history, start=1):
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "epoch",
            "train_loss",
            "validation_loss",
            "metrics",
        }:
            msg = "checkpoint history contains an unsupported epoch record"
            raise DeepTrainingError(msg)
        record_epoch = _checkpoint_int(raw_record["epoch"], field="history.epoch", minimum=1)
        if record_epoch != expected_epoch:
            msg = "checkpoint history epochs must be contiguous and one-indexed"
            raise DeepTrainingError(msg)
        raw_metrics = raw_record["metrics"]
        if not isinstance(raw_metrics, dict):
            msg = "checkpoint history metrics must be a mapping"
            raise DeepTrainingError(msg)
        metrics: dict[str, float] = {}
        for name, value in raw_metrics.items():
            if not isinstance(name, str) or not name:
                msg = "checkpoint history metric names must be non-empty strings"
                raise DeepTrainingError(msg)
            metrics[name] = _checkpoint_float(value, field=f"history.metrics.{name}")
        history.append(
            EpochRecord(
                epoch=record_epoch,
                train_loss=_checkpoint_float(raw_record["train_loss"], field="train_loss"),
                validation_loss=_checkpoint_float(
                    raw_record["validation_loss"], field="validation_loss"
                ),
                metrics=metrics,
            )
        )

    best_epoch = _checkpoint_int(loaded["best_epoch"], field="best_epoch", minimum=1)
    if best_epoch > epoch:
        msg = "checkpoint best_epoch cannot exceed the completed epoch"
        raise DeepTrainingError(msg)
    model_state = _checkpoint_state(loaded["model_state"], field="model_state")
    best_model_state = _checkpoint_state(loaded["best_model_state"], field="best_model_state")
    if set(model_state) != set(best_model_state):
        msg = "checkpoint current and best model state keys disagree"
        raise DeepTrainingError(msg)
    for name, tensor in model_state.items():
        best_tensor = best_model_state[name]
        if best_tensor.shape != tensor.shape or best_tensor.dtype != tensor.dtype:
            msg = f"checkpoint best_model_state tensor {name!r} disagrees in shape or dtype"
            raise DeepTrainingError(msg)

    optimizer_state = _checkpoint_optimizer_state(loaded["optimizer_state"])
    trainer_signature = _checkpoint_trainer_signature(loaded["trainer_signature"])
    model_signature = _checkpoint_model_signature(
        loaded["model_signature"], model_state=model_state
    )

    torch_rng_state = _checkpoint_rng_state(loaded["torch_rng_state"], field="torch_rng_state")
    shuffle_rng_state = _checkpoint_rng_state(
        loaded["shuffle_rng_state"], field="shuffle_rng_state"
    )
    best_validation_loss = _checkpoint_float(
        loaded["best_validation_loss"], field="best_validation_loss"
    )
    if best_validation_loss != history[best_epoch - 1].validation_loss:
        msg = "checkpoint best_validation_loss disagrees with the best history record"
        raise DeepTrainingError(msg)
    epochs_without_improvement = _checkpoint_int(
        loaded["epochs_without_improvement"],
        field="epochs_without_improvement",
    )
    if epochs_without_improvement != epoch - best_epoch:
        msg = "checkpoint early-stopping counter disagrees with best_epoch"
        raise DeepTrainingError(msg)

    return CheckpointPayload(
        schema=_CHECKPOINT_SCHEMA,
        epoch=epoch,
        model_state=model_state,
        optimizer_state=optimizer_state,
        torch_rng_state=torch_rng_state,
        shuffle_rng_state=shuffle_rng_state,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_model_state=best_model_state,
        epochs_without_improvement=epochs_without_improvement,
        trainer_signature=trainer_signature,
        model_signature=model_signature,
    )


def _batch_to_device(batch: Batch, device: torch.device) -> Batch:
    if not isinstance(batch, (tuple, list)) or not batch:
        msg = "data loader must yield a non-empty tensor sequence"
        raise DeepTrainingError(msg)
    moved: list[torch.Tensor] = []
    for index, tensor in enumerate(batch):
        if not isinstance(tensor, torch.Tensor):
            msg = f"data loader batch item {index} is not a torch.Tensor"
            raise DeepTrainingError(msg)
        moved.append(tensor.to(device))
    return tuple(moved)


def _finite_scalar(value: object, *, context: str) -> float:
    if not isinstance(value, torch.Tensor):
        msg = f"{context} must be returned as a torch.Tensor"
        raise DeepTrainingError(msg)
    if value.ndim != 0:
        msg = f"{context} must be a scalar tensor; received shape {tuple(value.shape)}"
        raise DeepTrainingError(msg)
    scalar = float(value.detach().cpu().item())
    if not math.isfinite(scalar):
        msg = f"{context} became non-finite; stopping instead of training on garbage"
        raise DeepTrainingError(msg)
    return scalar


def _require_nonempty_dataset(dataset: Dataset[Batch], *, context: str) -> None:
    if len(cast(Sized, dataset)) == 0:
        msg = f"{context} dataset must contain at least one example"
        raise DeepTrainingError(msg)


def _require_finite_model_state(model: nn.Module, *, context: str) -> None:
    for name, tensor in model.state_dict().items():
        if not bool(torch.all(torch.isfinite(tensor))):
            msg = f"{context} model state {name!r} became non-finite"
            raise DeepTrainingError(msg)


def _require_finite_gradients(model: nn.Module, *, epoch: int) -> None:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not bool(torch.all(torch.isfinite(gradient))):
            msg = f"gradient for parameter {name!r} became non-finite at epoch {epoch}"
            raise DeepTrainingError(msg)


def _objective_loss(
    objective: BatchObjective,
    model: nn.Module,
    batch: Batch,
    *,
    context: str,
) -> torch.Tensor:
    try:
        loss = objective.loss(model, batch)
    except DeepTrainingError:
        raise
    except Exception as error:
        msg = f"{context} objective failed: {error}"
        raise DeepTrainingError(msg) from error
    if not isinstance(loss, torch.Tensor):
        msg = f"{context} objective must return a torch.Tensor"
        raise DeepTrainingError(msg)
    return loss


def _objective_metrics(
    objective: Objective,
    model: nn.Module,
    batches: Iterator[Batch],
) -> dict[str, float]:
    try:
        raw_metrics = objective.evaluation_metrics(model, batches)
    except DeepTrainingError:
        raise
    except Exception as error:
        msg = f"evaluation metrics failed: {error}"
        raise DeepTrainingError(msg) from error
    if not isinstance(raw_metrics, Mapping):
        msg = "evaluation metrics must be a mapping"
        raise DeepTrainingError(msg)
    metrics: dict[str, float] = {}
    for name, value in raw_metrics.items():
        if not isinstance(name, str) or not name:
            msg = "evaluation metric names must be non-empty strings"
            raise DeepTrainingError(msg)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"evaluation metric {name!r} must be a real scalar"
            raise DeepTrainingError(msg)
        numeric = float(value)
        if not math.isfinite(numeric):
            msg = f"evaluation metric {name!r} is non-finite"
            raise DeepTrainingError(msg)
        metrics[name] = numeric
    return metrics


class Trainer:
    """One deterministic, resumable training loop for every PyTorch study."""

    def __init__(self, config: TrainerConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)

    @property
    def device(self) -> torch.device:
        """The resolved execution device."""

        return self._device

    def _trainer_signature(self) -> dict[str, object]:
        """Return resume-critical settings; the terminal epoch budget may grow."""

        return {
            "batch_size": self._config.batch_size,
            "learning_rate": self._config.learning_rate,
            "seed": self._config.seed,
            "optimizer": self._config.optimizer,
            "momentum": self._config.momentum,
            "weight_decay": self._config.weight_decay,
            "patience": self._config.patience,
            "min_delta": self._config.min_delta,
            "grad_clip_norm": self._config.grad_clip_norm,
            "device": self._config.device,
        }

    @staticmethod
    def _model_signature(model: nn.Module) -> dict[str, tuple[tuple[int, ...], str]]:
        return {
            name: (tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in model.state_dict().items()
        }

    # ----------------------------------------------------------------- loops

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self._config.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self._config.learning_rate,
                weight_decay=self._config.weight_decay,
            )
        return torch.optim.SGD(
            model.parameters(),
            lr=self._config.learning_rate,
            momentum=self._config.momentum,
            weight_decay=self._config.weight_decay,
        )

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create owned optimizer state for a paradigm-native interaction loop.

        The caller places the model on ``device`` before creating the optimizer.
        RL owns interaction/replay scheduling; the Trainer owns numerical updates.
        """

        _require_finite_model_state(model, context="initial model")
        return self._build_optimizer(model)

    def train_batch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        objective: BatchObjective,
        *,
        step: int,
    ) -> float:
        """One checked update shared by dataset epochs and RL minibatches.

        Reject invalid losses/gradients before stepping. An optimizer failure is
        fatal to the owned run; this method does not promise update rollback.
        ``step`` labels diagnostics and must be a positive integer.
        """

        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("step must be a positive integer")
        model.train()
        moved = _batch_to_device(batch, self._device)
        optimizer.zero_grad(set_to_none=True)
        loss = _objective_loss(objective, model, moved, context=f"training loss at epoch {step}")
        scalar = _finite_scalar(loss, context=f"training loss at epoch {step}")
        try:
            loss.backward()  # type: ignore[no-untyped-call]
        except RuntimeError as error:
            msg = f"gradient/backward pass failed at epoch {step}: {error}"
            raise DeepTrainingError(msg) from error
        _require_finite_gradients(model, epoch=step)
        if self._config.grad_clip_norm is not None:
            try:
                nn.utils.clip_grad_norm_(
                    model.parameters(), self._config.grad_clip_norm, error_if_nonfinite=True
                )
            except RuntimeError as error:
                msg = f"gradient clipping failed at epoch {step}: {error}"
                raise DeepTrainingError(msg) from error
        try:
            optimizer.step()
        except (RuntimeError, TypeError, ValueError) as error:
            msg = f"optimizer step failed at epoch {step}: {error}"
            raise DeepTrainingError(msg) from error
        _require_finite_model_state(model, context=f"optimizer step at epoch {step}")
        return scalar

    def _preflight_dataset(
        self,
        model: nn.Module,
        dataset: Dataset[Batch],
        objective: Objective,
        *,
        context: str,
    ) -> None:
        """Validate every batch before the first optimizer update."""

        loader = DataLoader(dataset, batch_size=self._config.batch_size, shuffle=False)
        original_state = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        original_rng = torch.get_rng_state()
        was_training = model.training
        model.eval()
        examples = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    moved = _batch_to_device(batch, self._device)
                    loss = _objective_loss(objective, model, moved, context=f"{context} preflight")
                    _finite_scalar(loss, context=f"{context} loss preflight")
                    examples += int(moved[0].shape[0])
            if examples == 0:
                msg = f"{context} dataset produced no batches during preflight"
                raise DeepTrainingError(msg)
            _require_finite_model_state(model, context=f"{context} preflight")
        finally:
            model.load_state_dict(original_state)
            model.train(was_training)
            torch.set_rng_state(original_rng)

    def _train_one_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader[Batch],
        objective: Objective,
        *,
        epoch: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            moved = _batch_to_device(batch, self._device)
            scalar = self.train_batch(model, optimizer, moved, objective, step=epoch)
            batch_examples = int(moved[0].shape[0])
            total_loss += scalar * batch_examples
            total_examples += batch_examples
        if total_examples == 0:
            msg = "training dataset produced no batches"
            raise DeepTrainingError(msg)
        return total_loss / total_examples

    def evaluate(
        self, model: nn.Module, dataset: Dataset[Batch], objective: Objective
    ) -> tuple[float, dict[str, float]]:
        """Return mean loss and objective metrics over a dataset without gradients."""

        _require_nonempty_dataset(dataset, context="evaluation")
        loader = DataLoader(dataset, batch_size=self._config.batch_size, shuffle=False)
        model = model.to(self._device)
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_examples = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    moved = _batch_to_device(batch, self._device)
                    loss = _objective_loss(
                        objective,
                        model,
                        moved,
                        context="evaluation loss",
                    )
                    scalar = _finite_scalar(loss, context="evaluation loss")
                    batch_examples = int(moved[0].shape[0])
                    total_loss += scalar * batch_examples
                    total_examples += batch_examples
                if total_examples == 0:
                    msg = "evaluation dataset produced no batches"
                    raise DeepTrainingError(msg)
                metric_batches = (
                    _batch_to_device(batch, self._device)
                    for batch in DataLoader(
                        dataset, batch_size=self._config.batch_size, shuffle=False
                    )
                )
                metrics = _objective_metrics(objective, model, metric_batches)
                _require_finite_model_state(model, context="evaluation")
        finally:
            model.train(was_training)
        return total_loss / total_examples, metrics

    def fit(
        self,
        model: nn.Module,
        train_dataset: Dataset[Batch],
        validation_dataset: Dataset[Batch],
        objective: Objective,
        *,
        checkpoint_path: Path | None = None,
        resume_from: Path | None = None,
    ) -> TrainingReport:
        """Train while restoring process-global Torch RNG/determinism state afterward."""

        previous_rng_state = torch.get_rng_state()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            return self._fit_owned(
                model,
                train_dataset,
                validation_dataset,
                objective,
                checkpoint_path=checkpoint_path,
                resume_from=resume_from,
            )
        finally:
            torch.set_rng_state(previous_rng_state)
            torch.use_deterministic_algorithms(
                previous_determinism,
                warn_only=previous_warn_only,
            )

    def _fit_owned(
        self,
        model: nn.Module,
        train_dataset: Dataset[Batch],
        validation_dataset: Dataset[Batch],
        objective: Objective,
        *,
        checkpoint_path: Path | None,
        resume_from: Path | None,
    ) -> TrainingReport:
        """Own optimizer/RNG/checkpoint state for one training execution."""

        _require_nonempty_dataset(train_dataset, context="training")
        _require_nonempty_dataset(validation_dataset, context="validation")
        _require_finite_model_state(model, context="initial model")
        payload: CheckpointPayload | None = None
        if resume_from is not None:
            payload = self.load_checkpoint(resume_from)
            if payload["trainer_signature"] != self._trainer_signature():
                msg = "checkpoint trainer configuration is incompatible with this run"
                raise DeepTrainingError(msg)
            if payload["model_signature"] != self._model_signature(model):
                msg = "checkpoint model signature is incompatible with this model"
                raise DeepTrainingError(msg)
            if payload["epoch"] >= self._config.max_epochs:
                msg = (
                    f"checkpoint already covers {payload['epoch']} epochs; "
                    f"max_epochs is {self._config.max_epochs}"
                )
                raise DeepTrainingError(msg)
            if (
                self._config.patience is not None
                and payload["epochs_without_improvement"] >= self._config.patience
            ):
                msg = (
                    "checkpoint records a terminal early-stopping state; "
                    "resume would execute an epoch that an uninterrupted run would not"
                )
                raise DeepTrainingError(msg)
            probe_optimizer = self._build_optimizer(model)
            _load_validated_optimizer_state(probe_optimizer, payload["optimizer_state"])

        model = model.to(self._device)
        optimizer = self._build_optimizer(model)
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(self._config.seed)
        torch.manual_seed(self._config.seed)

        history: list[EpochRecord] = []
        start_epoch = 0
        best_epoch = 0
        best_validation_loss = math.inf
        best_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }
        epochs_without_improvement = 0
        resumed_from_epoch: int | None = None

        if payload is not None:
            start_epoch = payload["epoch"]
            original_model_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
            original_torch_rng = torch.get_rng_state()
            original_shuffle_rng = shuffle_generator.get_state()
            try:
                # The optimizer is newly owned by this call, so validate and load
                # it before mutating the caller-owned model.
                optimizer.load_state_dict(payload["optimizer_state"])
                model.load_state_dict(payload["model_state"])
                torch.set_rng_state(payload["torch_rng_state"])
                shuffle_generator.set_state(payload["shuffle_rng_state"])
                _require_finite_model_state(model, context="restored checkpoint")
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                model.load_state_dict(original_model_state)
                torch.set_rng_state(original_torch_rng)
                shuffle_generator.set_state(original_shuffle_rng)
                msg = "checkpoint state cannot be restored into this model and optimizer"
                raise DeepTrainingError(msg) from error
            history = list(payload["history"])
            best_epoch = payload["best_epoch"]
            best_validation_loss = payload["best_validation_loss"]
            best_state = dict(payload["best_model_state"])
            epochs_without_improvement = payload["epochs_without_improvement"]
            resumed_from_epoch = start_epoch

        self._preflight_dataset(model, train_dataset, objective, context="training")
        self._preflight_dataset(model, validation_dataset, objective, context="validation")

        train_loader: DataLoader[Batch] = DataLoader(
            train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            generator=shuffle_generator,
            num_workers=0,
        )

        stopped_early = False
        for epoch in range(start_epoch + 1, self._config.max_epochs + 1):
            train_loss = self._train_one_epoch(
                model, optimizer, train_loader, objective, epoch=epoch
            )
            validation_loss, metrics = self.evaluate(model, validation_dataset, objective)
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    metrics=metrics,
                )
            )

            if best_validation_loss - validation_loss > self._config.min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if checkpoint_path is not None:
                self._save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    shuffle_generator=shuffle_generator,
                    history=tuple(history),
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                )

            if (
                self._config.patience is not None
                and epochs_without_improvement >= self._config.patience
            ):
                stopped_early = True
                break

        try:
            model.load_state_dict(best_state)
            _require_finite_model_state(model, context="best-weight restoration")
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            msg = "best model state could not be restored after training"
            raise DeepTrainingError(msg) from error
        return TrainingReport(
            history=tuple(history),
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            stopped_early=stopped_early,
            resumed_from_epoch=resumed_from_epoch,
        )

    # ----------------------------------------------------------- checkpoints

    def _save_checkpoint(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        shuffle_generator: torch.Generator,
        history: tuple[EpochRecord, ...],
        epoch: int,
        best_epoch: int,
        best_validation_loss: float,
        best_state: Mapping[str, torch.Tensor],
        epochs_without_improvement: int,
    ) -> None:
        _require_finite_model_state(model, context="checkpoint publication")
        optimizer_state = _checkpoint_optimizer_state(optimizer.state_dict())
        payload = {
            "schema": _CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "model_state": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "optimizer_state": optimizer_state,
            "torch_rng_state": torch.get_rng_state(),
            "shuffle_rng_state": shuffle_generator.get_state(),
            "history": tuple(
                {
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "validation_loss": record.validation_loss,
                    "metrics": dict(record.metrics),
                }
                for record in history
            ),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_model_state": dict(best_state),
            "epochs_without_improvement": epochs_without_improvement,
            "trainer_signature": self._trainer_signature(),
            "model_signature": self._model_signature(model),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            # Saving to a stable in-memory archive avoids embedding a randomized
            # temporary filename in PyTorch's ZIP container.
            serialized = BytesIO()
            torch.save(payload, serialized)
            temporary.write_bytes(serialized.getvalue())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def load_checkpoint(path: Path) -> CheckpointPayload:
        """Load and structurally validate a trainer checkpoint."""

        if not path.is_file():
            msg = f"checkpoint does not exist: {path}"
            raise DeepTrainingError(msg)
        try:
            loaded: object = torch.load(path, map_location="cpu", weights_only=True)
        except (
            EOFError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            msg = f"checkpoint is unreadable: {path}"
            raise DeepTrainingError(msg) from error
        if (
            not isinstance(loaded, dict)
            or not isinstance(loaded.get("schema"), str)
            or loaded.get("schema") != _CHECKPOINT_SCHEMA
        ):
            msg = f"checkpoint has an unsupported layout: {path}"
            raise DeepTrainingError(msg)
        if any(not isinstance(field, str) for field in loaded):
            msg = f"checkpoint contains unsupported non-string fields: {path}"
            raise DeepTrainingError(msg)
        missing = set(CheckpointPayload.__annotations__).difference(loaded)
        if missing:
            msg = f"checkpoint is missing fields: {', '.join(sorted(missing))}"
            raise DeepTrainingError(msg)
        unexpected = set(loaded).difference(CheckpointPayload.__annotations__)
        if unexpected:
            msg = f"checkpoint contains unsupported fields: {', '.join(sorted(unexpected))}"
            raise DeepTrainingError(msg)
        return _validated_checkpoint_payload(loaded, path=path)


# --------------------------------------------------------------------- objectives


def _classification_logits(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    if labels.ndim != 1:
        msg = f"{context} labels must be one-dimensional; received shape {tuple(labels.shape)}"
        raise DeepTrainingError(msg)
    if labels.dtype != torch.int64:
        msg = f"{context} labels must use torch.int64 class indices"
        raise DeepTrainingError(msg)
    if labels.shape[0] == 0:
        msg = f"{context} labels must be non-empty"
        raise DeepTrainingError(msg)
    try:
        logits = model(*inputs)
    except DeepTrainingError:
        raise
    except Exception as error:
        msg = f"{context} model forward failed: {error}"
        raise DeepTrainingError(msg) from error
    if not isinstance(logits, torch.Tensor):
        msg = f"{context} model must return a torch.Tensor"
        raise DeepTrainingError(msg)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[1] < 2:
        msg = (
            f"{context} logits must have shape (n, classes>=2) matching labels; "
            f"received {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
        raise DeepTrainingError(msg)
    if not logits.is_floating_point() or logits.is_complex():
        msg = f"{context} logits must be real floating-point values"
        raise DeepTrainingError(msg)
    if logits.device != labels.device:
        msg = f"{context} logits and labels must be on the same device"
        raise DeepTrainingError(msg)
    if not bool(torch.all(torch.isfinite(logits))):
        msg = f"{context} logits contain non-finite values"
        raise DeepTrainingError(msg)
    if int(torch.min(labels).item()) < 0 or int(torch.max(labels).item()) >= logits.shape[1]:
        msg = f"{context} labels must lie in [0, {logits.shape[1] - 1}]"
        raise DeepTrainingError(msg)
    return logits


def _reconstruction_output(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    try:
        reconstructed = model(inputs)
    except DeepTrainingError:
        raise
    except Exception as error:
        msg = f"reconstruction model forward failed: {error}"
        raise DeepTrainingError(msg) from error
    if not isinstance(reconstructed, torch.Tensor):
        msg = "reconstruction model must return a torch.Tensor"
        raise DeepTrainingError(msg)
    if reconstructed.shape != inputs.shape:
        msg = (
            "reconstruction output must exactly match the input shape; "
            f"received {tuple(reconstructed.shape)} and {tuple(inputs.shape)}"
        )
        raise DeepTrainingError(msg)
    if not reconstructed.is_floating_point() or reconstructed.is_complex():
        msg = "reconstruction output must contain real floating-point values"
        raise DeepTrainingError(msg)
    if reconstructed.device != inputs.device:
        msg = "reconstruction output and input must be on the same device"
        raise DeepTrainingError(msg)
    if not bool(torch.all(torch.isfinite(reconstructed))):
        msg = "reconstruction output contains non-finite values"
        raise DeepTrainingError(msg)
    return reconstructed


class ClassificationObjective:
    """Cross-entropy on ``(inputs, labels)`` batches with accuracy evidence."""

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        inputs, labels = _expect_arity(batch, expected=2, context="classification")
        logits = _classification_logits(model, (inputs,), labels, context="classification")
        return nn.functional.cross_entropy(logits, labels)

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        correct = 0
        total = 0
        for batch in batches:
            inputs, labels = _expect_arity(batch, expected=2, context="classification")
            predictions = _classification_logits(
                model,
                (inputs,),
                labels,
                context="classification",
            ).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
        if total == 0:
            msg = "classification evaluation received no examples"
            raise DeepTrainingError(msg)
        return {"accuracy": correct / total}


class SequenceClassificationObjective:
    """Cross-entropy on ``(padded, lengths, labels)`` variable-length batches."""

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        padded, lengths, labels = _expect_arity(batch, expected=3, context="sequence")
        logits = _classification_logits(
            model,
            (padded, lengths),
            labels,
            context="sequence",
        )
        return nn.functional.cross_entropy(logits, labels)

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        correct = 0
        total = 0
        for batch in batches:
            padded, lengths, labels = _expect_arity(batch, expected=3, context="sequence")
            predictions = _classification_logits(
                model,
                (padded, lengths),
                labels,
                context="sequence",
            ).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
        if total == 0:
            msg = "sequence evaluation received no examples"
            raise DeepTrainingError(msg)
        return {"accuracy": correct / total}


class ReconstructionObjective:
    """Mean-squared reconstruction error on ``(inputs,)`` batches."""

    def loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        (inputs,) = _expect_arity(batch, expected=1, context="reconstruction")
        return nn.functional.mse_loss(_reconstruction_output(model, inputs), inputs)

    def evaluation_metrics(self, model: nn.Module, batches: Iterator[Batch]) -> dict[str, float]:
        del model, batches
        return {}


def _expect_arity(batch: Batch, *, expected: int, context: str) -> Batch:
    if len(batch) != expected:
        msg = f"{context} objective expects {expected}-tensor batches; received {len(batch)}"
        raise DeepTrainingError(msg)
    return batch


def tensor_dataset(*tensors: torch.Tensor) -> TensorDataset:
    """Build a :class:`TensorDataset` after validating a shared sample dimension."""

    if not tensors:
        msg = "tensor_dataset requires at least one tensor"
        raise ValueError(msg)
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            msg = f"tensor_dataset item {index} is not a torch.Tensor"
            raise TypeError(msg)
        if tensor.ndim == 0:
            msg = f"tensor_dataset item {index} must have a sample dimension"
            raise ValueError(msg)
        if tensor.shape[0] == 0:
            msg = "tensor_dataset requires at least one sample"
            raise ValueError(msg)
        if tensor.is_complex():
            msg = f"tensor_dataset item {index} must be real-valued"
            raise TypeError(msg)
        if tensor.is_floating_point() and not bool(torch.all(torch.isfinite(tensor))):
            msg = f"tensor_dataset item {index} contains non-finite values"
            raise ValueError(msg)
    sizes = {int(tensor.shape[0]) for tensor in tensors}
    if len(sizes) != 1:
        msg = f"tensors disagree on sample count: {sorted(sizes)}"
        raise ValueError(msg)
    return TensorDataset(*tensors)
