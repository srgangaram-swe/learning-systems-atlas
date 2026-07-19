"""Supervised classification validity, selection, and serialization tests."""

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import ClassificationBenchmarkConfig
from learning_atlas.core.contracts import LearningParadigm, RunContext
from learning_atlas.supervised.classification import (
    ClassificationBenchmark,
    build_candidates,
    load_classification_data,
)

pytestmark = pytest.mark.unit


def quick_config() -> ClassificationBenchmarkConfig:
    return ClassificationBenchmarkConfig(cv_folds=2, forest_estimators=10, forest_max_depth=4)


def test_stratified_split_precedes_preprocessing_and_is_reproducible() -> None:
    first = load_classification_data(seed=42, test_size=0.2)
    second = load_classification_data(seed=42, test_size=0.2)
    np.testing.assert_array_equal(first.x_train, second.x_train)
    assert len(first.x_train) == 455
    assert len(first.x_test) == 114
    assert np.mean(first.y_train) == pytest.approx(np.mean(first.y_test), abs=0.01)

    logistic = build_candidates(quick_config())["logistic_regression"]
    logistic.fit(first.x_train, first.y_train)
    fitted_mean = logistic.named_steps["scale"].mean_
    np.testing.assert_allclose(fitted_mean, first.x_train.mean(axis=0))
    assert not np.allclose(fitted_mean, np.vstack((first.x_train, first.x_test)).mean(axis=0))


def test_benchmark_beats_prior_and_serializes_pipeline(tmp_path: Path) -> None:
    config = quick_config()
    data = load_classification_data(config.seed, config.test_size)
    result = ClassificationBenchmark(config).run(
        RunContext(seed=config.seed, artifacts=ArtifactStore(tmp_path))
    )

    assert result.paradigm is LearningParadigm.SUPERVISED
    assert (
        result.selected_model
        == max(
            result.candidates,
            key=lambda candidate: candidate.metrics["cv_roc_auc_mean"],
        ).name
    )
    assert result.selected_model != "dummy_prior"
    assert result.metrics["test_roc_auc"] > 0.95
    assert result.metrics["roc_auc_gain_vs_dummy"] > 0.40
    assert {candidate.name for candidate in result.candidates} == {
        "dummy_prior",
        "logistic_regression",
        "random_forest",
    }

    model = joblib.load(tmp_path / result.artifacts["model"])
    probability = model.predict_proba(data.x_test)[:, 1]
    assert roc_auc_score(data.y_test, probability) == pytest.approx(
        result.metrics["test_roc_auc"], abs=1e-12
    )
    assert (tmp_path / result.artifacts["diagnostics_plot"]).stat().st_size > 1_000
    assert (tmp_path / result.artifacts["model_comparison_plot"]).stat().st_size > 1_000
