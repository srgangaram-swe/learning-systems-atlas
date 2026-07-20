"""A multilayer perceptron trained end to end on the from-scratch autograd engine."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.estimators import ClassifierMixin, Estimator
from learning_atlas.core.reproducibility import derive_named_seed, generator_for_seed
from learning_atlas.core.validation import FloatArray, validate_choices
from learning_atlas.deep.autograd import AutogradError, Tensor

ActivationName = Literal["relu", "tanh"]

_ACTIVATIONS = ("relu", "tanh")


class MLPTrainingError(RuntimeError):
    """Raised when optimization degenerates instead of returning silent garbage."""

    def __init__(self, epoch: int, detail: str) -> None:
        self.epoch = epoch
        super().__init__(f"MLP training failed at epoch {epoch}: {detail}")


def _validated_hidden_layers(hidden_layer_sizes: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(hidden_layer_sizes)
    if not layers:
        msg = "hidden_layer_sizes must contain at least one layer"
        raise ValueError(msg)
    for width in layers:
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            msg = "hidden_layer_sizes must contain positive integers"
            raise ValueError(msg)
    return layers


def _validated_unit_interval(value: float, *, name: str, inclusive_zero: bool) -> float:
    numeric = float(value)
    lower_ok = numeric >= 0.0 if inclusive_zero else numeric > 0.0
    if not np.isfinite(numeric) or not lower_ok or numeric >= 1.0:
        bound = "[0, 1)" if inclusive_zero else "(0, 1)"
        msg = f"{name} must be a finite value in {bound}"
        raise ValueError(msg)
    return numeric


def _validated_positive(value: float, *, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        msg = f"{name} must be a finite positive value"
        raise ValueError(msg)
    return numeric


def _validated_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _stable_softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return np.asarray(exponent / np.sum(exponent, axis=1, keepdims=True), dtype=np.float64)


class MLPClassifier(ClassifierMixin, Estimator):
    """Softmax-output perceptron with minibatch SGD, momentum, and seeded init.

    All gradients flow through :mod:`learning_atlas.deep.autograd`; no external
    learning framework participates in either the forward or the backward pass.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (16,),
        *,
        activation: ActivationName = "tanh",
        learning_rate: float = 0.3,
        momentum: float = 0.9,
        batch_size: int = 32,
        max_epochs: int = 300,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._hidden_layer_sizes = _validated_hidden_layers(hidden_layer_sizes)
        self._activation = cast(
            ActivationName,
            validate_choices(activation, name="activation", choices=_ACTIVATIONS),
        )
        self._learning_rate = _validated_positive(learning_rate, name="learning_rate")
        self._momentum = _validated_unit_interval(momentum, name="momentum", inclusive_zero=True)
        self._batch_size = _validated_positive_int(batch_size, name="batch_size")
        self._max_epochs = _validated_positive_int(max_epochs, name="max_epochs")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            msg = "seed must be an integer in [0, 2**32 - 1]"
            raise ValueError(msg)
        self._seed = seed
        self._weights: list[Tensor] = []
        self._biases: list[Tensor] = []
        self._classes: FloatArray | None = None
        self._loss_history: tuple[float, ...] = ()

    # ---------------------------------------------------------------- learned state

    @property
    def classes_(self) -> FloatArray:
        """Sorted class labels observed during fitting."""

        self._require_fitted()
        assert self._classes is not None
        classes = self._classes.copy()
        classes.setflags(write=False)
        return classes

    @property
    def loss_history_(self) -> tuple[float, ...]:
        """Mean training cross-entropy per epoch."""

        self._require_fitted()
        return self._loss_history

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        """All trainable tensors, alternating layer weights and biases."""

        self._require_fitted()
        return self._all_parameters()

    # ---------------------------------------------------------------------- fitting

    def _validated_class_indices(self, targets: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
        if np.any(np.mod(targets, 1.0) != 0.0):
            msg = "classification targets must be integer class labels"
            raise ValueError(msg)
        classes = np.unique(targets)
        if len(classes) < 2:
            msg = "classification requires at least two distinct classes"
            raise ValueError(msg)
        return classes, np.asarray(np.searchsorted(classes, targets), dtype=np.int64)

    def _initialize_parameters(self, n_features: int, n_classes: int) -> None:
        rng = generator_for_seed(derive_named_seed(self._seed, "mlp", "initialization"))
        sizes = (n_features, *self._hidden_layer_sizes, n_classes)
        self._weights = []
        self._biases = []
        for fan_in, fan_out in pairwise(sizes):
            gain = 2.0 if self._activation == "relu" else 1.0
            scale = np.sqrt(gain / fan_in)
            weight = rng.normal(0.0, scale, size=(fan_in, fan_out))
            self._weights.append(Tensor(weight, requires_grad=True))
            self._biases.append(Tensor(np.zeros((1, fan_out)), requires_grad=True))

    def _forward(self, batch: Tensor) -> Tensor:
        hidden = batch
        final_layer = len(self._weights) - 1
        for index, (weight, bias) in enumerate(zip(self._weights, self._biases, strict=True)):
            hidden = hidden.matmul(weight) + bias
            if index != final_layer:
                hidden = hidden.relu() if self._activation == "relu" else hidden.tanh()
        return hidden

    def _batch_cross_entropy(
        self, features: FloatArray, class_indices: NDArray[np.int64]
    ) -> Tensor:
        logits = self._forward(Tensor(features))
        log_probabilities = logits.log_softmax(axis=1)
        one_hot = np.zeros(logits.shape, dtype=np.float64)
        one_hot[np.arange(len(class_indices)), class_indices] = 1.0
        picked = log_probabilities * Tensor(one_hot)
        return -picked.sum() / float(len(class_indices))

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Train with shuffled minibatch SGD plus classical momentum."""

        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes, class_indices = self._validated_class_indices(validated_targets)
        previous = (
            self._weights,
            self._biases,
            self._classes,
            self._loss_history,
            self._n_features_in,
        )
        try:
            self._initialize_parameters(validated_features.shape[1], len(classes))
            velocities = [np.zeros_like(parameter.data) for parameter in self._all_parameters()]
            shuffle_rng = generator_for_seed(
                derive_named_seed(self._seed, "mlp", "minibatch-shuffle")
            )

            history: list[float] = []
            n_samples = len(validated_features)
            for epoch in range(1, self._max_epochs + 1):
                order = shuffle_rng.permutation(n_samples)
                for start in range(0, n_samples, self._batch_size):
                    batch_index = order[start : start + self._batch_size]
                    try:
                        loss = self._batch_cross_entropy(
                            validated_features[batch_index], class_indices[batch_index]
                        )
                        for parameter in self._all_parameters():
                            parameter.zero_grad()
                        loss.backward()
                    except AutogradError as error:
                        raise MLPTrainingError(epoch, str(error)) from error
                    for index, (parameter, velocity) in enumerate(
                        zip(self._all_parameters(), velocities, strict=True)
                    ):
                        gradient = parameter.grad
                        assert gradient is not None  # loss depends on every parameter
                        if not np.all(np.isfinite(gradient)):
                            detail = f"parameter {index} gradient became non-finite"
                            raise MLPTrainingError(epoch, detail)
                        with np.errstate(over="ignore", invalid="ignore"):
                            candidate_velocity = self._momentum * velocity - (
                                self._learning_rate * gradient
                            )
                            candidate_parameter = parameter.data + candidate_velocity
                        if not np.all(np.isfinite(candidate_velocity)):
                            detail = f"parameter {index} momentum buffer became non-finite"
                            raise MLPTrainingError(epoch, detail)
                        if not np.all(np.isfinite(candidate_parameter)):
                            detail = f"parameter {index} update became non-finite"
                            raise MLPTrainingError(epoch, detail)
                        velocity[...] = candidate_velocity
                        parameter.data = candidate_parameter
                mean_loss = self._batch_cross_entropy(validated_features, class_indices).item()
                if not np.isfinite(mean_loss):
                    raise MLPTrainingError(epoch, "mean training loss became non-finite")
                history.append(float(mean_loss))

            classes = classes.copy()
            classes.setflags(write=False)
            self._classes = classes
            self._loss_history = tuple(history)
            self._mark_fitted(validated_features.shape[1])
        except BaseException:
            (
                self._weights,
                self._biases,
                self._classes,
                self._loss_history,
                self._n_features_in,
            ) = previous
            raise
        return self

    def _all_parameters(self) -> tuple[Tensor, ...]:
        combined: list[Tensor] = []
        for weight, bias in zip(self._weights, self._biases, strict=True):
            combined.extend((weight, bias))
        return tuple(combined)

    # -------------------------------------------------------------------- inference

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return class-membership probabilities in ``classes_`` order."""

        validated = self._validate_predict_data(features)
        logits = self._forward(Tensor(validated))
        return _stable_softmax(logits.data)

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return the most probable class label per sample."""

        probabilities = self.predict_proba(features)
        assert self._classes is not None
        return np.asarray(self._classes[np.argmax(probabilities, axis=1)], dtype=np.float64)

    def batch_loss(self, features: ArrayLike, targets: ArrayLike) -> Tensor:
        """Return the differentiable cross-entropy of one batch at the current weights.

        This exposes the live computation graph so gradient-check evidence can
        compare ``backward`` against finite differences on a frozen minibatch.
        """

        self._require_fitted()
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        assert self._classes is not None
        if not np.all(np.isin(validated_targets, self._classes)):
            msg = "batch_loss targets must use fitted class labels"
            raise ValueError(msg)
        class_indices = np.searchsorted(self._classes, validated_targets)
        return self._batch_cross_entropy(validated_features, class_indices)
