"""Transactional runner and manifest-integrity tests."""

import hashlib
import json
from pathlib import Path

import pytest

from learning_atlas.core.config import RegressionBenchmarkConfig
from learning_atlas.core.contracts import Experiment, LearningParadigm, RunContext, RunResult
from learning_atlas.core.runner import run_experiment

pytestmark = pytest.mark.unit


class SuccessfulExperiment:
    def __init__(self, result: RunResult) -> None:
        self._result = result

    def run(self, context: RunContext) -> RunResult:
        context.artifacts.write_text("models/fixture.txt", "model")
        return self._result.model_copy(update={"artifacts": {"model": "models/fixture.txt"}})


class FailingExperiment:
    def run(self, context: RunContext) -> RunResult:
        context.artifacts.write_text("partial.txt", "partial")
        raise RuntimeError("experiment failed")


class OverriddenResultExperiment:
    def __init__(self, result: RunResult, update: dict[str, object]) -> None:
        self._result = result
        self._update = update

    def run(self, context: RunContext) -> RunResult:
        context.artifacts.write_text("models/fixture.txt", "model")
        return self._result.model_copy(update=self._update)


def test_runner_publishes_complete_checksums(
    tmp_path: Path,
    sample_result: RunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment: Experiment = SuccessfulExperiment(sample_result)
    monkeypatch.setattr("learning_atlas.core.runner.build_experiment", lambda _config: experiment)
    output = tmp_path / "run"

    result = run_experiment(RegressionBenchmarkConfig(), output)

    assert result.artifacts["manifest"] == "manifest.json"
    assert not any(path.name.startswith(".run.") for path in tmp_path.iterdir())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0.0"
    physical_files = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["artifacts_sha256"]) == physical_files
    for relative, expected_digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected_digest
    assert RunResult.model_validate_json((output / "result.json").read_text()) == result
    assert all((output / relative).is_file() for relative in result.artifacts.values())


def test_runner_refuses_to_overwrite_non_empty_directory(
    tmp_path: Path,
    sample_result: RunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "learning_atlas.core.runner.build_experiment",
        lambda _config: SuccessfulExperiment(sample_result),
    )
    output = tmp_path / "run"
    run_experiment(RegressionBenchmarkConfig(), output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_experiment(RegressionBenchmarkConfig(), output)


def test_failed_run_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "learning_atlas.core.runner.build_experiment", lambda _config: FailingExperiment()
    )
    output = tmp_path / "failed"

    with pytest.raises(RuntimeError, match="experiment failed"):
        run_experiment(RegressionBenchmarkConfig(), output)

    assert not output.exists()
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"experiment": "clustering_benchmark"}, "experiment does not match"),
        ({"seed": 7}, "seed does not match"),
        ({"paradigm": LearningParadigm.UNSUPERVISED}, "paradigm does not match"),
        ({"selected_model": "missing"}, "selected_model"),
        ({"metrics": {"score": 0.6}}, "headline metric diverges"),
        ({"artifacts": {"model": "missing.bin"}}, "does not exist"),
        ({"artifacts": {"manifest": "models/fixture.txt"}}, "reserved artifact keys"),
        ({"artifacts": {"model": "./result.json"}}, "reserved artifact"),
        (
            {
                "artifacts": {
                    "first": "models/fixture.txt",
                    "second": "models/./fixture.txt",
                }
            },
            "paths must be unique",
        ),
        ({"artifacts": {"model": "../escape.bin"}}, "run directory"),
    ],
)
def test_runner_rejects_inconsistent_results_and_cleans_staging(
    tmp_path: Path,
    sample_result: RunResult,
    monkeypatch: pytest.MonkeyPatch,
    update: dict[str, object],
    message: str,
) -> None:
    experiment = OverriddenResultExperiment(sample_result, update)
    monkeypatch.setattr("learning_atlas.core.runner.build_experiment", lambda _config: experiment)
    output = tmp_path / "invalid"

    with pytest.raises(ValueError, match=message):
        run_experiment(RegressionBenchmarkConfig(), output)

    assert not output.exists()
    assert tuple(tmp_path.iterdir()) == ()
