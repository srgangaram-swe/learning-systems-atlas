"""Mathematical identities, edge contracts, and oracle agreement for metrics."""

import numpy as np
import pytest
from sklearn import metrics as sklearn_metrics

from learning_atlas.supervised.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

pytestmark = pytest.mark.unit


def test_regression_metrics_match_hand_computed_fixture() -> None:
    observed = [1.0, 2.0, 3.0]
    predicted = [1.0, 1.0, 4.0]

    assert mean_squared_error(observed, predicted) == pytest.approx(2.0 / 3.0)
    assert root_mean_squared_error(observed, predicted) == pytest.approx(np.sqrt(2.0 / 3.0))
    assert mean_absolute_error(observed, predicted) == pytest.approx(2.0 / 3.0)
    assert r2_score(observed, predicted) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(6))
def test_regression_metric_invariants_hold_across_deterministic_samples(seed: int) -> None:
    generator = np.random.default_rng(seed)
    observed = generator.normal(size=40)
    predicted = generator.normal(size=40)

    assert mean_squared_error(observed, predicted) >= 0.0
    assert root_mean_squared_error(observed, predicted) >= 0.0
    assert mean_absolute_error(observed, predicted) >= 0.0
    assert mean_squared_error(observed, predicted) == pytest.approx(
        mean_squared_error(predicted, observed)
    )
    assert mean_absolute_error(observed, predicted) == pytest.approx(
        mean_absolute_error(predicted, observed)
    )
    assert mean_squared_error(observed, observed) == 0.0
    assert mean_absolute_error(observed, observed) == 0.0
    assert r2_score(observed, observed) == 1.0


def test_r2_constant_target_convention_is_finite() -> None:
    assert r2_score([4.0, 4.0], [4.0, 4.0]) == 1.0
    assert r2_score([4.0, 4.0], [3.0, 3.0]) == 0.0


def test_classification_metrics_match_hand_computed_multiclass_fixture() -> None:
    observed = [0, 1, 2, 2]
    predicted = [0, 2, 2, 1]

    np.testing.assert_array_equal(
        confusion_matrix(observed, predicted),
        [[1, 0, 0], [0, 0, 1], [0, 1, 1]],
    )
    assert accuracy_score(observed, predicted) == 0.5
    assert precision_score(observed, predicted) == pytest.approx(0.5)
    assert recall_score(observed, predicted) == pytest.approx(0.5)
    assert f1_score(observed, predicted) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(5))
def test_classification_metrics_agree_with_sklearn_oracle(seed: int) -> None:
    generator = np.random.default_rng(seed)
    observed = generator.integers(0, 4, size=100)
    predicted = generator.integers(0, 4, size=100)

    np.testing.assert_array_equal(
        confusion_matrix(observed, predicted),
        sklearn_metrics.confusion_matrix(observed, predicted),
    )
    assert accuracy_score(observed, predicted) == pytest.approx(
        sklearn_metrics.accuracy_score(observed, predicted)
    )
    assert precision_score(observed, predicted) == pytest.approx(
        sklearn_metrics.precision_score(observed, predicted, average="macro", zero_division=0)
    )
    assert recall_score(observed, predicted) == pytest.approx(
        sklearn_metrics.recall_score(observed, predicted, average="macro", zero_division=0)
    )
    assert f1_score(observed, predicted) == pytest.approx(
        sklearn_metrics.f1_score(observed, predicted, average="macro", zero_division=0)
    )
    for metric in (accuracy_score, precision_score, recall_score, f1_score):
        assert 0.0 <= metric(observed, predicted) <= 1.0


def test_explicit_label_order_and_zero_division_are_respected() -> None:
    observed = [1, 1, 1]
    predicted = [1, 1, 1]
    np.testing.assert_array_equal(
        confusion_matrix(observed, predicted, labels=[2, 1]),
        [[0, 0], [0, 3]],
    )
    assert precision_score(observed, predicted, labels=[2, 1]) == 0.5
    assert recall_score(observed, predicted, labels=[2, 1], zero_division=1.0) == 1.0
    assert f1_score(observed, predicted, labels=[2, 1], zero_division=1.0) == 1.0


def test_binary_log_loss_is_clipped_and_matches_oracle() -> None:
    observed = np.asarray([0, 1, 1, 0])
    probability = np.asarray([0.0, 1.0, 0.8, 0.2])
    result = log_loss(observed, probability)

    assert np.isfinite(result)
    assert result == pytest.approx(
        sklearn_metrics.log_loss(observed, probability, labels=[0, 1]), abs=1e-12
    )
    assert log_loss(observed, 1.0 - probability, labels=[1, 0]) == pytest.approx(result)


def test_multiclass_log_loss_matches_oracle_with_nonstandard_labels() -> None:
    observed = np.asarray([10, 30, 20, 10])
    labels = [10, 20, 30]
    probability = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.7, 0.1],
            [0.7, 0.1, 0.2],
        ]
    )
    assert log_loss(observed, probability, labels=labels) == pytest.approx(
        sklearn_metrics.log_loss(observed, probability, labels=labels), abs=1e-12
    )


@pytest.mark.parametrize(
    ("observed", "predicted", "exception", "message"),
    [
        ([], [], ValueError, "at least one"),
        ([[1.0]], [1.0], ValueError, "observed targets must be 1D"),
        ([1.0], [[1.0]], ValueError, "targets must be 1D"),
        ([1.0, 2.0], [1.0], ValueError, "inconsistent"),
        ([1.0], [np.nan], ValueError, "finite"),
        ([1.0], ["one"], TypeError, "real numeric"),
    ],
)
def test_paired_metric_validation_is_actionable(
    observed: object,
    predicted: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        mean_squared_error(observed, predicted)


def test_classification_label_validation_rejects_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        confusion_matrix([0], [0], labels=[[0]])
    with pytest.raises(ValueError, match="at least one class"):
        confusion_matrix([0], [0], labels=[])
    with pytest.raises(ValueError, match="duplicates"):
        confusion_matrix([0], [0], labels=[0, 0])
    with pytest.raises(ValueError, match="absent"):
        confusion_matrix([0, 1], [0, 2], labels=[0, 1])
    with pytest.raises(ValueError, match="zero_division"):
        precision_score([0], [0], zero_division=-1.0)
    with pytest.raises(ValueError, match="average"):
        precision_score([0], [0], average="binary")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="average"):
        recall_score([0], [0], average="binary")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="average"):
        f1_score([0], [0], average="binary")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("observed", "probability", "kwargs", "exception", "message"),
    [
        ([], [], {}, ValueError, "at least one"),
        ([[0]], [0.5], {}, ValueError, "observed targets must be 1D"),
        ([0, 1], [0.5], {}, ValueError, "inconsistent"),
        ([0, 1], [0.5, np.nan], {}, ValueError, "finite"),
        ([0, 1], [0.5, 1.1], {}, ValueError, r"\[0, 1\]"),
        ([0, 1], ["no", "yes"], {}, TypeError, "real numeric"),
        ([0, 1], [[[0.5, 0.5]]], {}, ValueError, "1D or 2D"),
        ([0, 1], [[0.5], [0.5]], {}, ValueError, "at least two"),
        ([0, 1], [[0.8, 0.3], [0.2, 0.8]], {}, ValueError, "sum to 1"),
        ([0, 1], [0.2, 0.8], {"epsilon": 0.5}, ValueError, "epsilon"),
        ([0, 1, 2], [0.2, 0.8, 0.4], {}, ValueError, "exactly two"),
        ([0, 1], [0.2, 0.8], {"labels": [0, 0]}, ValueError, "exactly two"),
        ([0, 2], [0.2, 0.8], {"labels": [0, 1]}, ValueError, "absent"),
        ([0, 1], [0.2, 0.8], {"labels": 1}, ValueError, "labels must be 1D"),
        ([0, 1], [[0.2], [0.3, 0.7]], {}, ValueError, "rectangular"),
        ([0, 1], [[0.8, 0.2]], {}, ValueError, "inconsistent"),
        ([0, 1], [[0.8, 0.2], [0.2, 0.8]], {"labels": [[0, 1]]}, ValueError, "1D"),
        ([0, 1], [[0.8, 0.2], [0.2, 0.8]], {"labels": [0, 1, 2]}, ValueError, "number"),
        ([0, 1], [[0.8, 0.2], [0.2, 0.8]], {"labels": [0, 0]}, ValueError, "duplicates"),
        ([0, 2], [[0.8, 0.2], [0.2, 0.8]], {"labels": [0, 1]}, ValueError, "absent"),
    ],
)
def test_log_loss_rejects_invalid_probability_contracts(
    observed: object,
    probability: object,
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        log_loss(observed, probability, **kwargs)  # type: ignore[arg-type]
