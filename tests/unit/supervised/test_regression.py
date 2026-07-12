"""Supervised benchmark validity and serialization tests."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import RegressionBenchmarkConfig
from learning_atlas.core.contracts import LearningParadigm, RunContext
from learning_atlas.supervised.regression import (
    RegressionBenchmark,
    build_candidates,
    load_regression_data,
)

pytestmark = pytest.mark.unit


def quick_config() -> RegressionBenchmarkConfig:
    return RegressionBenchmarkConfig(cv_folds=2, forest_estimators=10, forest_max_depth=4)


def test_split_precedes_preprocessing_and_is_reproducible() -> None:
    first = load_regression_data(seed=42, test_size=0.2)
    second = load_regression_data(seed=42, test_size=0.2)
    np.testing.assert_array_equal(first.x_train, second.x_train)
    assert len(first.x_train) == 353
    assert len(first.x_test) == 89

    ridge = build_candidates(quick_config())["ridge"]
    ridge.fit(first.x_train, first.y_train)
    fitted_mean = ridge.named_steps["scale"].mean_
    np.testing.assert_allclose(fitted_mean, first.x_train.mean(axis=0))
    assert not np.allclose(fitted_mean, np.vstack((first.x_train, first.x_test)).mean(axis=0))


def test_benchmark_beats_dummy_and_serializes_pipeline(tmp_path: Path) -> None:
    context = RunContext(seed=42, artifacts=ArtifactStore(tmp_path))
    result = RegressionBenchmark(quick_config()).run(context)

    assert result.paradigm is LearningParadigm.SUPERVISED
    assert result.selected_model in {"ridge", "random_forest"}
    assert (
        result.selected_model
        == min(
            result.candidates,
            key=lambda candidate: candidate.metrics["cv_rmse_mean"],
        ).name
    )
    assert result.metrics["rmse_reduction_vs_dummy"] > 0.15
    assert result.metrics["test_r2"] > 0.30
    assert {candidate.name for candidate in result.candidates} == {
        "dummy_mean",
        "ridge",
        "random_forest",
    }
    model = joblib.load(tmp_path / result.artifacts["model"])
    prediction = model.predict(load_regression_data(42, 0.2).x_test[:3])
    assert prediction.shape == (3,)
    assert (tmp_path / result.artifacts["diagnostics_plot"]).stat().st_size > 1_000
    assert (tmp_path / result.artifacts["model_comparison_plot"]).stat().st_size > 1_000
