"""Naive baseline fitted-state and prediction tests."""

import numpy as np
import pytest

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.baselines import MeanRegressor, PriorClassifier

pytestmark = pytest.mark.unit


def test_mean_regressor_predicts_training_mean() -> None:
    features = np.arange(8, dtype=np.float64).reshape(4, 2)
    model = MeanRegressor().fit(features, [1.0, 3.0, 5.0, 7.0])
    assert model.mean_ == 4.0
    np.testing.assert_array_equal(model.predict(features[:2]), [4.0, 4.0])


def test_prior_classifier_probabilities_and_deterministic_tie() -> None:
    features = np.arange(12, dtype=np.float64).reshape(6, 2)
    model = PriorClassifier().fit(features, [3, 1, 3, 1, 3, 1])
    np.testing.assert_array_equal(model.classes_, [1.0, 3.0])
    np.testing.assert_allclose(model.predict_proba(features[:2]), [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_array_equal(model.predict(features[:2]), [1.0, 1.0])


def test_baselines_reject_invalid_state_and_single_class() -> None:
    with pytest.raises(NotFittedError):
        MeanRegressor().predict([[1.0]])
    with pytest.raises(NotFittedError):
        _ = PriorClassifier().class_prior_
    with pytest.raises(ValueError, match="two target classes"):
        PriorClassifier().fit([[0.0], [1.0]], [4.0, 4.0])
