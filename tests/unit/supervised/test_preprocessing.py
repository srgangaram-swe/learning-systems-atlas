"""Leakage, determinism, round-trip, and failure tests for preprocessing."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.preprocessing import StandardScaler, train_test_split

pytestmark = pytest.mark.unit


def test_standard_scaler_matches_oracle_and_round_trips() -> None:
    features = np.random.default_rng(3).normal(
        loc=[4.0, -3.0, 9.0], scale=[2.0, 5.0, 0.5], size=(80, 3)
    )
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)

    np.testing.assert_allclose(transformed, SklearnStandardScaler().fit_transform(features))
    np.testing.assert_allclose(np.mean(transformed, axis=0), 0.0, atol=3e-15)
    np.testing.assert_allclose(np.std(transformed, axis=0), 1.0, atol=1e-15)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), features, atol=1e-14)
    np.testing.assert_allclose(scaler.mean_, np.mean(features, axis=0))
    np.testing.assert_allclose(scaler.var_, np.var(features, axis=0))
    assert scaler.n_features_in_ == 3


def test_standard_scaler_handles_constant_features_without_nan() -> None:
    features = np.asarray([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)

    np.testing.assert_array_equal(scaler.scale_, [np.std([1.0, 2.0, 3.0]), 1.0])
    np.testing.assert_array_equal(transformed[:, 1], 0.0)
    assert np.all(np.isfinite(transformed))
    np.testing.assert_allclose(scaler.inverse_transform(transformed), features)


@pytest.mark.parametrize(
    ("with_mean", "with_std", "expected"),
    [
        (False, True, [[1.0, 2.0], [3.0, 4.0]]),
        (True, False, [[-1.0, -1.0], [1.0, 1.0]]),
        (False, False, [[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_scaler_configuration_is_invertible(
    with_mean: bool,
    with_std: bool,
    expected: list[list[float]],
) -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    transformed = StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(features)
    if not with_std:
        np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_allclose(
        StandardScaler(with_mean=with_mean, with_std=with_std)
        .fit(features)
        .inverse_transform(transformed),
        features,
    )


def test_scaler_state_is_defensive_and_fit_failure_does_not_corrupt_prior_state() -> None:
    scaler = StandardScaler().fit([[1.0], [3.0]])
    exported_mean = scaler.mean_
    exported_mean[0] = 999.0
    assert scaler.mean_[0] == 2.0

    with pytest.raises(ValueError, match="finite"):
        scaler.fit([[np.nan]])
    assert scaler.n_features_in_ == 1
    np.testing.assert_array_equal(scaler.transform([[2.0]]), [[0.0]])


def test_scaler_rejects_inference_before_fit_and_feature_drift() -> None:
    scaler = StandardScaler()
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = scaler.mean_
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = scaler.var_
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = scaler.scale_
    with pytest.raises(NotFittedError, match="not fitted"):
        _ = scaler.n_features_in_
    with pytest.raises(NotFittedError, match="not fitted"):
        scaler.transform([[1.0]])
    with pytest.raises(ValueError, match="expects 2"):
        StandardScaler().fit([[1.0, 2.0]]).transform([[1.0]])
    with pytest.raises(TypeError, match="booleans"):
        StandardScaler(with_mean=1)  # type: ignore[arg-type]


def test_train_test_split_is_repeatable_complete_and_tuple_compatible() -> None:
    features = np.arange(60, dtype=np.float64).reshape(20, 3)
    targets = np.arange(20, dtype=np.float64)
    first = train_test_split(features, targets, test_size=0.2, seed=17)
    second = train_test_split(features, targets, test_size=0.2, seed=17)
    x_train, x_test, y_train, y_test = first

    np.testing.assert_array_equal(first.x_train, second.x_train)
    np.testing.assert_array_equal(first.x_test, second.x_test)
    assert x_train.shape == (16, 3)
    assert x_test.shape == (4, 3)
    assert y_train.shape == (16,)
    assert y_test.shape == (4,)
    assert set(np.concatenate((y_train, y_test))) == set(targets)
    np.testing.assert_array_equal(features, np.arange(60, dtype=np.float64).reshape(20, 3))


def test_integer_test_size_and_nonshuffled_order_are_explicit() -> None:
    features = np.arange(12, dtype=np.float64).reshape(6, 2)
    targets = np.arange(6)
    split = train_test_split(features, targets, test_size=2, shuffle=False)

    np.testing.assert_array_equal(split.x_test, features[:2])
    np.testing.assert_array_equal(split.y_test, targets[:2])
    np.testing.assert_array_equal(split.x_train, features[2:])


def test_stratification_preserves_each_class_and_requested_partition_size() -> None:
    features = np.arange(200, dtype=np.float64).reshape(100, 2)
    targets = np.repeat([10.0, 20.0, 30.0], [50, 30, 20])
    split = train_test_split(
        features,
        targets,
        test_size=0.23,
        seed=9,
        stratify=targets,
    )

    assert len(split.y_test) == 23
    assert len(split.y_train) == 77
    assert set(np.unique(split.y_test)) == {10.0, 20.0, 30.0}
    assert set(np.unique(split.y_train)) == {10.0, 20.0, 30.0}
    for label, expected_fraction in ((10.0, 0.5), (20.0, 0.3), (30.0, 0.2)):
        assert np.mean(split.y_test == label) == pytest.approx(expected_fraction, abs=0.04)


def test_generator_state_advances_but_integer_seed_replays() -> None:
    features = np.arange(40, dtype=np.float64).reshape(20, 2)
    targets = np.arange(20)
    generator = np.random.default_rng(4)
    first = train_test_split(features, targets, seed=generator)
    second = train_test_split(features, targets, seed=generator)
    replay = train_test_split(features, targets, seed=4)

    np.testing.assert_array_equal(first.x_test, replay.x_test)
    assert not np.array_equal(first.x_test, second.x_test)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"test_size": 0.0}, ValueError, "strictly between"),
        ({"test_size": 1.0}, ValueError, "strictly between"),
        ({"test_size": np.nan}, ValueError, "strictly between"),
        ({"test_size": 0}, ValueError, "integer"),
        ({"test_size": 10}, ValueError, "integer"),
        ({"test_size": True}, TypeError, "float proportion"),
        ({"test_size": "small"}, TypeError, "float proportion"),
        ({"shuffle": 1}, TypeError, "boolean"),
    ],
)
def test_split_rejects_invalid_partition_parameters(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        train_test_split(np.ones((10, 2)), np.arange(10), **kwargs)  # type: ignore[arg-type]


def test_split_rejects_impossible_or_invalid_stratification() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        train_test_split([[1.0]], [1.0])
    with pytest.raises(ValueError, match="at least two classes"):
        train_test_split(np.ones((6, 1)), np.ones(6), stratify=np.ones(6))
    with pytest.raises(ValueError, match="at least two samples"):
        train_test_split(
            np.ones((6, 1)),
            np.arange(6),
            stratify=[0, 0, 0, 0, 0, 1],
        )
    with pytest.raises(ValueError, match="each have at least one"):
        train_test_split(
            np.ones((6, 1)),
            np.arange(6),
            test_size=1,
            stratify=[0, 0, 0, 1, 1, 1],
        )
    with pytest.raises(ValueError, match="inconsistent"):
        train_test_split(np.ones((6, 1)), np.arange(6), stratify=[0, 1])
