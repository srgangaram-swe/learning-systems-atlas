"""Estimator, optimization, reproducibility, and learning tests for the scratch MLP."""

import pickle
from collections.abc import Callable

import numpy as np
import pytest

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.deep.autograd import Tensor, finite_difference_gradient
from learning_atlas.deep.datasets import PlanarDataset, make_two_moons, make_xor
from learning_atlas.deep.mlp import MLPClassifier, MLPTrainingError
from learning_atlas.supervised.preprocessing import train_test_split

pytestmark = pytest.mark.unit


def _binary_training_data() -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([[-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5]])
    targets = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return features, targets


def _fitted_state_snapshot(
    model: MLPClassifier,
    features: np.ndarray,
) -> tuple[
    int,
    np.ndarray,
    tuple[float, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
]:
    return (
        model.n_features_in_,
        model.classes_,
        model.loss_history_,
        tuple(parameter.data.copy() for parameter in model.parameters),
        model.predict(features),
        model.predict_proba(features),
    )


def _assert_fitted_state_matches(
    model: MLPClassifier,
    features: np.ndarray,
    snapshot: tuple[
        int,
        np.ndarray,
        tuple[float, ...],
        tuple[np.ndarray, ...],
        np.ndarray,
        np.ndarray,
    ],
) -> None:
    n_features, classes, history, parameters, predictions, probabilities = snapshot
    assert model.is_fitted is True
    assert model.n_features_in_ == n_features
    np.testing.assert_array_equal(model.classes_, classes)
    assert model.loss_history_ == history
    np.testing.assert_array_equal(model.predict(features), predictions)
    np.testing.assert_array_equal(model.predict_proba(features), probabilities)
    for expected, actual in zip(parameters, model.parameters, strict=True):
        np.testing.assert_array_equal(actual.data, expected)


def _one_epoch_fitted_model(features: np.ndarray, targets: np.ndarray) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(3,),
        activation="tanh",
        learning_rate=0.3,
        momentum=0.9,
        batch_size=len(features),
        max_epochs=1,
        seed=71,
    ).fit(features, targets)


def _inject_after_backward(
    monkeypatch: pytest.MonkeyPatch,
    model: MLPClassifier,
    mutation: Callable[[Tensor], None],
) -> None:
    original_backward = Tensor.backward

    def backward_with_injection(tensor: Tensor, gradient: object | None = None) -> None:
        if gradient is None:
            original_backward(tensor)
        else:
            original_backward(tensor, gradient)  # type: ignore[arg-type]
        mutation(model._all_parameters()[0])

    monkeypatch.setattr(Tensor, "backward", backward_with_injection)


@pytest.mark.parametrize(
    "access",
    [
        lambda model: model.classes_,
        lambda model: model.loss_history_,
        lambda model: model.parameters,
        lambda model: model.predict([[0.0]]),
        lambda model: model.predict_proba([[0.0]]),
        lambda model: model.batch_loss([[0.0]], [0.0]),
    ],
)
def test_unfitted_state_is_rejected_consistently(access: Callable[[MLPClassifier], object]) -> None:
    with pytest.raises(NotFittedError, match="not fitted"):
        access(MLPClassifier(hidden_layer_sizes=(2,), max_epochs=2))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: MLPClassifier(hidden_layer_sizes=()), "at least one"),
        (lambda: MLPClassifier(hidden_layer_sizes=(4, 0)), "positive integers"),
        (lambda: MLPClassifier(hidden_layer_sizes=(True,)), "positive integers"),
        (lambda: MLPClassifier(activation="sigmoid"), "activation"),  # type: ignore[arg-type]
        (lambda: MLPClassifier(learning_rate=0.0), "learning_rate"),
        (lambda: MLPClassifier(learning_rate=np.nan), "learning_rate"),
        (lambda: MLPClassifier(momentum=-0.1), "momentum"),
        (lambda: MLPClassifier(momentum=1.0), "momentum"),
        (lambda: MLPClassifier(batch_size=0), "batch_size"),
        (lambda: MLPClassifier(batch_size=True), "batch_size"),
        (lambda: MLPClassifier(max_epochs=0), "max_epochs"),
        (lambda: MLPClassifier(max_epochs=True), "max_epochs"),
        (lambda: MLPClassifier(seed=True), "seed"),
        (lambda: MLPClassifier(seed=-1), "seed"),
        (lambda: MLPClassifier(seed=2**32), "seed"),
    ],
)
def test_invalid_hyperparameters_fail_before_training(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_multiclass_fit_predict_and_probability_contracts() -> None:
    generator = np.random.default_rng(5)
    features = np.vstack(
        [
            generator.normal((-2.0, -1.5), 0.15, size=(24, 2)),
            generator.normal((2.0, -1.0), 0.15, size=(24, 2)),
            generator.normal((0.0, 2.0), 0.15, size=(24, 2)),
        ]
    )
    targets = np.repeat(np.asarray([-3.0, 4.0, 9.0]), 24)
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        batch_size=12,
        max_epochs=80,
        seed=7,
    ).fit(features, targets)

    probabilities = model.predict_proba(features)

    np.testing.assert_array_equal(model.classes_, [-3.0, 4.0, 9.0])
    assert probabilities.shape == (len(features), 3)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-15)
    assert model.score(features, targets) >= 0.98
    assert model.n_features_in_ == 2


def test_exported_classes_are_defensive() -> None:
    features, targets = _binary_training_data()
    model = MLPClassifier(hidden_layer_sizes=(3,), max_epochs=20, seed=4).fit(features, targets)

    exposed = model.classes_
    assert exposed.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        exposed[:] = 999.0

    np.testing.assert_array_equal(model.classes_, [0.0, 1.0])


def test_training_replays_exactly_and_does_not_use_process_global_rng() -> None:
    features, targets = _binary_training_data()
    np.random.seed(1729)
    global_before = np.random.get_state()

    first = MLPClassifier(
        hidden_layer_sizes=(4, 3),
        batch_size=2,
        max_epochs=35,
        seed=23,
    ).fit(features, targets)
    global_after = np.random.get_state()
    second = MLPClassifier(
        hidden_layer_sizes=(4, 3),
        batch_size=2,
        max_epochs=35,
        seed=23,
    ).fit(features, targets)

    assert first.loss_history_ == second.loss_history_
    for left, right in zip(first.parameters, second.parameters, strict=True):
        np.testing.assert_array_equal(left.data, right.data)
    assert global_before[0] == global_after[0]
    assert global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(global_before[1], global_after[1])


@pytest.mark.parametrize(
    ("factory", "seed"),
    [
        (lambda seed: make_xor(160, noise=0.15, seed=seed), 11),
        (lambda seed: make_two_moons(160, noise=0.1, seed=seed), 12),
    ],
    ids=("xor", "two-moons"),
)
def test_mlp_learns_nonlinear_planar_tasks(
    factory: Callable[[int], PlanarDataset], seed: int
) -> None:
    dataset = factory(seed)
    split = train_test_split(
        dataset.features,
        dataset.targets,
        test_size=0.25,
        seed=21,
        stratify=dataset.targets,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="tanh",
        learning_rate=0.3,
        momentum=0.9,
        batch_size=16,
        max_epochs=120,
        seed=31,
    ).fit(split.x_train, split.y_train)

    assert model.score(split.x_train, split.y_train) >= 0.98
    assert model.score(split.x_test, split.y_test) >= 0.95
    assert model.loss_history_[-1] < 0.05 * model.loss_history_[0]
    assert len(model.loss_history_) == 120
    assert np.all(np.isfinite(model.loss_history_))


def test_recorded_epoch_loss_matches_full_training_objective_after_final_update() -> None:
    features, targets = _binary_training_data()
    model = MLPClassifier(
        hidden_layer_sizes=(3,),
        batch_size=4,
        max_epochs=12,
        seed=8,
    ).fit(features, targets)

    observed = model.batch_loss(features, targets).item()

    assert model.loss_history_[-1] == pytest.approx(observed, rel=1e-12, abs=1e-12)


def test_frozen_minibatch_gradient_matches_finite_differences() -> None:
    features, targets = _binary_training_data()
    model = MLPClassifier(
        hidden_layer_sizes=(3,),
        batch_size=3,
        max_epochs=20,
        seed=13,
    ).fit(features, targets)
    frozen_features = features[:4]
    frozen_targets = targets[:4]
    loss = model.batch_loss(frozen_features, frozen_targets)
    for parameter in model.parameters:
        parameter.zero_grad()
    loss.backward()

    errors: list[float] = []
    for parameter in model.parameters:
        assert parameter.grad is not None
        analytic = parameter.grad.copy()
        numeric = finite_difference_gradient(
            lambda: model.batch_loss(frozen_features, frozen_targets),
            parameter,
        )
        errors.append(float(np.max(np.abs(analytic - numeric))))

    assert max(errors) <= 1e-5


def test_pickle_round_trip_preserves_state_history_and_inference() -> None:
    features, targets = _binary_training_data()
    model = MLPClassifier(hidden_layer_sizes=(3,), max_epochs=25, seed=14).fit(features, targets)

    restored = pickle.loads(pickle.dumps(model, protocol=5))

    np.testing.assert_array_equal(restored.classes_, model.classes_)
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))
    np.testing.assert_array_equal(restored.predict_proba(features), model.predict_proba(features))
    assert restored.loss_history_ == model.loss_history_
    for left, right in zip(restored.parameters, model.parameters, strict=True):
        np.testing.assert_array_equal(left.data, right.data)


def test_nonfinite_gradient_guard_is_contextual_and_refit_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, targets = _binary_training_data()
    model = _one_epoch_fitted_model(features, targets)
    snapshot = _fitted_state_snapshot(model, features)

    def inject_nonfinite_gradient(parameter: Tensor) -> None:
        assert parameter.grad is not None
        parameter.grad.fill(np.nan)

    _inject_after_backward(monkeypatch, model, inject_nonfinite_gradient)

    with pytest.raises(
        MLPTrainingError,
        match=r"epoch 1: parameter 0 gradient became non-finite",
    ):
        model.fit(features, targets)

    _assert_fitted_state_matches(model, features, snapshot)


def test_nonfinite_momentum_guard_is_contextual_and_refit_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, targets = _binary_training_data()
    model = _one_epoch_fitted_model(features, targets)
    snapshot = _fitted_state_snapshot(model, features)
    monkeypatch.setattr(model, "_learning_rate", 2.0)

    def inject_extreme_finite_gradient(parameter: Tensor) -> None:
        assert parameter.grad is not None
        parameter.grad.fill(np.finfo(np.float64).max)

    _inject_after_backward(monkeypatch, model, inject_extreme_finite_gradient)

    with pytest.raises(
        MLPTrainingError,
        match=r"epoch 1: parameter 0 momentum buffer became non-finite",
    ):
        model.fit(features, targets)

    _assert_fitted_state_matches(model, features, snapshot)


def test_nonfinite_parameter_update_guard_is_contextual_and_refit_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, targets = _binary_training_data()
    model = _one_epoch_fitted_model(features, targets)
    snapshot = _fitted_state_snapshot(model, features)

    def inject_overflowing_finite_update(parameter: Tensor) -> None:
        assert parameter.grad is not None
        parameter.data.fill(np.finfo(np.float64).max)
        parameter.grad.fill(-np.finfo(np.float64).max)

    _inject_after_backward(monkeypatch, model, inject_overflowing_finite_update)

    with pytest.raises(
        MLPTrainingError,
        match=r"epoch 1: parameter 0 update became non-finite",
    ):
        model.fit(features, targets)

    _assert_fitted_state_matches(model, features, snapshot)


def test_nonfinite_epoch_mean_loss_guard_is_contextual_and_refit_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, targets = _binary_training_data()
    model = _one_epoch_fitted_model(features, targets)
    snapshot = _fitted_state_snapshot(model, features)
    original_batch_cross_entropy = model._batch_cross_entropy
    calls = 0

    def corrupt_only_epoch_mean_loss(
        batch_features: np.ndarray,
        class_indices: np.ndarray,
    ) -> Tensor:
        nonlocal calls
        calls += 1
        loss = original_batch_cross_entropy(batch_features, class_indices)
        if calls == 2:
            loss.data[...] = np.nan
        return loss

    monkeypatch.setattr(model, "_batch_cross_entropy", corrupt_only_epoch_mean_loss)

    with pytest.raises(
        MLPTrainingError,
        match=r"epoch 1: mean training loss became non-finite",
    ):
        model.fit(features, targets)

    assert calls == 2
    _assert_fitted_state_matches(model, features, snapshot)


def test_failed_refit_rolls_back_every_piece_of_previous_fitted_state() -> None:
    features, targets = _binary_training_data()
    model = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="relu",
        batch_size=2,
        max_epochs=3,
        seed=0,
    ).fit(features, targets)
    predictions_before = model.predict(features)
    probabilities_before = model.predict_proba(features)
    classes_before = model.classes_
    history_before = model.loss_history_
    parameters_before = tuple(parameter.data.copy() for parameter in model.parameters)
    extreme_features = np.asarray(
        [
            [1e308, 1e308],
            [1e308, 1e308],
            [-1e308, -1e308],
            [-1e308, -1e308],
            [1e308, 1e308],
            [-1e308, -1e308],
        ]
    )

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(MLPTrainingError, match="epoch"):
            model.fit(extreme_features, targets)

    assert model.is_fitted is True
    assert model.n_features_in_ == 1
    np.testing.assert_array_equal(model.classes_, classes_before)
    assert model.loss_history_ == history_before
    np.testing.assert_array_equal(model.predict(features), predictions_before)
    np.testing.assert_array_equal(model.predict_proba(features), probabilities_before)
    for expected, actual in zip(parameters_before, model.parameters, strict=True):
        np.testing.assert_array_equal(actual.data, expected)


def test_fit_and_batch_loss_reject_invalid_targets_and_feature_shapes() -> None:
    features, targets = _binary_training_data()
    model = MLPClassifier(hidden_layer_sizes=(2,), max_epochs=5, seed=2)

    with pytest.raises(ValueError, match="integer class labels"):
        model.fit(features, targets + 0.25)
    with pytest.raises(ValueError, match="at least two"):
        model.fit(features, np.zeros(len(features)))
    with pytest.raises(ValueError, match="2D"):
        model.fit(features[:, 0], targets)

    fitted = model.fit(features, targets)
    with pytest.raises(ValueError, match="fitted class labels"):
        fitted.batch_loss(features[:2], [0.0, 7.0])
    with pytest.raises(ValueError, match="expects 1"):
        fitted.predict([[0.0, 1.0]])
