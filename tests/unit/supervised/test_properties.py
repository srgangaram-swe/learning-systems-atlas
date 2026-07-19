"""Property-based numerical invariants required by the Sprint 2 quality contract."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learning_atlas.supervised.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from learning_atlas.supervised.preprocessing import StandardScaler

pytestmark = pytest.mark.unit
PROPERTY_SETTINGS = settings(max_examples=60, derandomize=True, database=None, deadline=None)
FINITE = st.floats(min_value=-1.0e4, max_value=1.0e4, allow_nan=False, allow_infinity=False)
CLASS_LABEL = st.integers(min_value=-4, max_value=4)


@PROPERTY_SETTINGS
@given(st.lists(st.tuples(FINITE, FINITE), min_size=1, max_size=40))
def test_regression_metric_invariants(pairs: list[tuple[float, float]]) -> None:
    observed = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    predicted = np.asarray([pair[1] for pair in pairs], dtype=np.float64)

    mse = mean_squared_error(observed, predicted)
    rmse = root_mean_squared_error(observed, predicted)
    assert mse >= 0.0
    assert mean_absolute_error(observed, predicted) >= 0.0
    assert rmse >= 0.0
    assert rmse * rmse == pytest.approx(mse, rel=1e-12, abs=1e-12)
    assert mean_squared_error(observed, predicted) == pytest.approx(
        mean_squared_error(predicted, observed)
    )
    assert mean_absolute_error(observed, predicted) == pytest.approx(
        mean_absolute_error(predicted, observed)
    )


@PROPERTY_SETTINGS
@given(st.lists(FINITE, min_size=2, max_size=40))
def test_perfect_prediction_identities(values: list[float]) -> None:
    observed = np.asarray(values, dtype=np.float64)
    assert mean_squared_error(observed, observed) == 0.0
    assert root_mean_squared_error(observed, observed) == 0.0
    assert mean_absolute_error(observed, observed) == 0.0
    assert r2_score(observed, observed) == 1.0
    assert accuracy_score(observed, observed) == 1.0


@PROPERTY_SETTINGS
@given(st.lists(st.tuples(CLASS_LABEL, CLASS_LABEL), min_size=1, max_size=80))
def test_classification_metric_invariants(pairs: list[tuple[int, int]]) -> None:
    observed = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    predicted = np.asarray([pair[1] for pair in pairs], dtype=np.float64)

    scores = (
        accuracy_score(observed, predicted),
        precision_score(observed, predicted),
        recall_score(observed, predicted),
        f1_score(observed, predicted),
    )
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert int(np.sum(confusion_matrix(observed, predicted))) == len(observed)

    perfect_scores = (
        accuracy_score(observed, observed),
        precision_score(observed, observed),
        recall_score(observed, observed),
        f1_score(observed, observed),
    )
    assert perfect_scores == (1.0, 1.0, 1.0, 1.0)
    perfect_confusion = confusion_matrix(observed, observed)
    assert int(np.trace(perfect_confusion)) == len(observed)
    assert int(np.sum(perfect_confusion) - np.trace(perfect_confusion)) == 0


@PROPERTY_SETTINGS
@given(
    st.lists(
        st.tuples(
            st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
            st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
            st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=2,
        max_size=30,
    )
)
def test_standard_scaler_round_trip(rows: list[tuple[float, float, float]]) -> None:
    features = np.asarray(rows, dtype=np.float64)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    reconstructed = scaler.inverse_transform(transformed)

    assert np.all(np.isfinite(transformed))
    np.testing.assert_allclose(reconstructed, features, rtol=1e-11, atol=1e-11)
