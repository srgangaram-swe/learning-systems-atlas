"""End-to-end supervised pipelines and cross-solver numerical agreement."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pytestmark = pytest.mark.integration


def test_regression_pipeline_beats_naive_and_ridge_solvers_agree() -> None:
    features, target = make_regression(
        n_samples=240,
        n_features=8,
        n_informative=8,
        noise=0.5,
        random_state=17,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=23,
    )
    predictions = []
    for solver in ("svd", "cholesky"):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver=solver))
        predictions.append(model.fit(x_train, y_train).predict(x_test))

    naive = np.full_like(y_test, np.mean(y_train))
    assert all(r2_score(y_test, prediction) > 0.99 for prediction in predictions)
    assert all(
        r2_score(y_test, prediction) > r2_score(y_test, naive) + 0.50 for prediction in predictions
    )
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-10, atol=1e-10)


def test_classification_pipeline_beats_naive_and_logistic_solvers_agree() -> None:
    features, target = make_classification(
        n_samples=500,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        class_sep=2.0,
        random_state=19,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=29,
        stratify=target,
    )
    probabilities = []
    accuracies = []
    for solver in ("lbfgs", "liblinear"):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver=solver, max_iter=2_000, random_state=31),
        )
        model.fit(x_train, y_train)
        probabilities.append(model.predict_proba(x_test)[:, 1])
        accuracies.append(accuracy_score(y_test, model.predict(x_test)))

    naive_accuracy = max(float(np.mean(y_test)), 1.0 - float(np.mean(y_test)))
    assert all(accuracy > naive_accuracy + 0.25 for accuracy in accuracies)
    assert abs(accuracies[0] - accuracies[1]) <= 0.02
    assert np.corrcoef(probabilities[0], probabilities[1])[0, 1] > 0.999
