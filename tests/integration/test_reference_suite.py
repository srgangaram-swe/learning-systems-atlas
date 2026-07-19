"""Cross-paradigm suite, reproducibility, and serialization tests."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from learning_atlas.core.config import RegressionBenchmarkConfig, load_config
from learning_atlas.core.contracts import RunResult
from learning_atlas.core.runner import run_experiment
from learning_atlas.supervised.regression import load_regression_data
from learning_atlas.workflows.suite import discover_configs, run_suite

pytestmark = pytest.mark.integration


def write_quick_configs(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "regression.yaml").write_text(
        "experiment: regression_benchmark\nseed: 42\ncv_folds: 2\n"
        "forest_estimators: 10\nforest_max_depth: 4\n",
        encoding="utf-8",
    )
    (root / "classification.yaml").write_text(
        "experiment: classification_benchmark\nseed: 42\ncv_folds: 2\n"
        "forest_estimators: 10\nforest_max_depth: 4\n",
        encoding="utf-8",
    )
    (root / "clustering.yaml").write_text(
        "experiment: clustering_benchmark\nseed: 42\nn_samples: 200\nkmeans_n_init: 3\n",
        encoding="utf-8",
    )
    (root / "q_learning.yaml").write_text(
        "experiment: q_learning_frozen_lake\nseed: 42\nis_slippery: false\n"
        "training_episodes: 1500\nevaluation_episodes: 100\nepsilon_decay: 0.995\n",
        encoding="utf-8",
    )


def test_reference_suite_runs_all_paradigms_and_round_trips_models(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    write_quick_configs(configs)
    output = tmp_path / "suite"

    results = run_suite(configs, output)

    assert {result.paradigm.value for result in results} == {
        "supervised",
        "unsupervised",
        "reinforcement",
    }
    assert len(discover_configs(configs)) == 4
    for result in results:
        run_dir = output / result.experiment
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["config_sha256"]
        assert manifest["environment"]["git"]
        assert set(manifest["artifacts_sha256"]) >= {
            "resolved_config.json",
            "result.json",
        }

    regression = next(result for result in results if result.experiment == "regression_benchmark")
    fitted = joblib.load(output / regression.experiment / regression.artifacts["model"])
    held_out = load_regression_data(seed=42, test_size=0.2)
    predictions = fitted.predict(held_out.x_test)
    observed_rmse = float(np.sqrt(np.mean((held_out.y_test - predictions) ** 2)))
    assert observed_rmse == pytest.approx(regression.metrics["test_rmse"], abs=1e-12)

    reinforcement = next(
        result for result in results if result.experiment == "q_learning_frozen_lake"
    )
    q_values = np.load(output / reinforcement.experiment / reinforcement.artifacts["q_table"])[
        "q_values"
    ]
    assert q_values.shape == (16, 4)


def test_same_seed_reproduces_regression_metrics(tmp_path: Path) -> None:
    config = RegressionBenchmarkConfig(cv_folds=2, forest_estimators=10, forest_max_depth=4)
    first = run_experiment(config, tmp_path / "first")
    second = run_experiment(config, tmp_path / "second")

    assert first == second
    first_manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    second_manifest = json.loads((tmp_path / "second/manifest.json").read_text())
    assert first_manifest["artifacts_sha256"] == second_manifest["artifacts_sha256"]
    for volatile in ("created_at", "duration_seconds"):
        first_manifest.pop(volatile)
        second_manifest.pop(volatile)
    assert first_manifest == second_manifest


def test_suite_rejects_missing_empty_and_duplicate_configs(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="does not exist"):
        discover_configs(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no YAML"):
        discover_configs(empty)

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    content = "experiment: regression_benchmark\ncv_folds: 2\nforest_estimators: 10\n"
    (duplicate / "one.yaml").write_text(content)
    (duplicate / "two.yml").write_text(content)
    duplicate_output = tmp_path / "duplicate-output"
    with pytest.raises(ValueError, match="duplicate"):
        run_suite(duplicate, duplicate_output)
    assert not duplicate_output.exists()


def test_suite_preflight_and_execution_failures_publish_nothing(
    tmp_path: Path,
    sample_result: RunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = tmp_path / "configs"
    write_quick_configs(configs)
    (configs / "z-invalid.yaml").write_text("experiment: [", encoding="utf-8")
    invalid_output = tmp_path / "invalid-output"
    with pytest.raises(ValueError, match="invalid YAML"):
        run_suite(configs, invalid_output)
    assert not invalid_output.exists()
    (configs / "z-invalid.yaml").unlink()

    call_count = 0

    def fail_second_run(_config: object, output: Path) -> RunResult:
        nonlocal call_count
        call_count += 1
        output.mkdir(parents=True)
        (output / "partial.txt").write_text("partial")
        if call_count == 2:
            raise RuntimeError("second experiment failed")
        return sample_result

    monkeypatch.setattr("learning_atlas.workflows.suite.run_experiment", fail_second_run)
    failed_output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="second experiment"):
        run_suite(configs, failed_output)
    assert not failed_output.exists()
    assert not tuple(tmp_path.glob(".failed-output.*"))


def test_committed_slippery_rl_profile_has_clear_effect(tmp_path: Path) -> None:
    config = load_config(Path("configs/reinforcement/q_learning.yaml"))
    result = run_experiment(config, tmp_path / "rl")
    candidates = {candidate.name: candidate for candidate in result.candidates}
    learned = candidates["q_learning"].metrics
    random = candidates["random_policy"].metrics

    assert learned["success_rate"] >= 0.65
    assert learned["improvement_vs_random"] >= 0.60
    assert learned["training_last_100_mean_return"] >= 0.35
    assert learned["success_ci95_low"] > random["success_ci95_high"]
