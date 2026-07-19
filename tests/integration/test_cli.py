"""Command-line behavior tests through Typer's public runner."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import learning_atlas.cli as cli_module
from learning_atlas import __version__
from learning_atlas.cli import app
from learning_atlas.supervised.comparison import CandidateExecutionError
from learning_atlas.unsupervised.comparison import UnsupervisedCandidateError

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_discovery_version_schema_and_validation() -> None:
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert version.stdout.strip() == __version__

    discovered = runner.invoke(app, ["list"])
    assert discovered.exit_code == 0
    assert "regression_benchmark\tsupervised" in discovered.stdout
    assert "classification_benchmark\tsupervised" in discovered.stdout
    assert "scratch_regression_benchmark\tsupervised" in discovered.stdout
    assert "scratch_classification_benchmark\tsupervised" in discovered.stdout
    assert "scratch_clustering_benchmark\tunsupervised" in discovered.stdout
    assert "scratch_representation_benchmark\tunsupervised" in discovered.stdout
    assert "q_learning_frozen_lake\treinforcement" in discovered.stdout

    schema = runner.invoke(app, ["schema"])
    assert schema.exit_code == 0
    assert "clustering_benchmark" in schema.stdout
    assert "ClusteringBenchmarkConfig" in json.loads(schema.stdout)["$defs"]
    assert "ScratchClusteringBenchmarkConfig" in json.loads(schema.stdout)["$defs"]
    assert "ScratchRepresentationBenchmarkConfig" in json.loads(schema.stdout)["$defs"]

    valid = runner.invoke(app, ["validate", "configs/supervised/regression.yaml"])
    assert valid.exit_code == 0
    assert json.loads(valid.stdout)["experiment"] == "regression_benchmark"


def test_invalid_config_returns_actionable_exit(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("experiment: regression_benchmark\nsead: 42\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(invalid)])
    assert result.exit_code == 2
    assert "invalid configuration" in result.stderr
    assert "sead" in result.stderr

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("experiment: [", encoding="utf-8")
    malformed_result = runner.invoke(app, ["validate", str(malformed)])
    assert malformed_result.exit_code == 2
    assert "invalid YAML configuration" in malformed_result.stderr


def test_run_command_publishes_result(tmp_path: Path) -> None:
    config = tmp_path / "regression.yaml"
    config.write_text(
        "\n".join(
            (
                "experiment: regression_benchmark",
                "seed: 42",
                "cv_folds: 2",
                "forest_estimators: 10",
                "forest_max_depth: 4",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    result = runner.invoke(app, ["run", str(config), "--output-dir", str(output)])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["paradigm"] == "supervised"
    assert (output / "manifest.json").is_file()
    rerun = runner.invoke(app, ["run", str(config), "--output-dir", str(output)])
    assert rerun.exit_code == 2
    assert "refusing to overwrite" in rerun.stderr


def test_supervised_candidate_failure_returns_actionable_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_benchmark(_config_dir: Path, _output_dir: Path) -> None:
        raise CandidateExecutionError("rbf_svm", "CV fold 2")

    monkeypatch.setattr(cli_module, "run_supervised_benchmark", fail_benchmark)
    result = runner.invoke(
        app,
        [
            "benchmark-supervised",
            "--config-dir",
            str(tmp_path / "configs"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 2
    assert "supervised benchmark failed" in result.stderr
    assert "rbf_svm" in result.stderr
    assert "CV fold 2" in result.stderr


def test_unsupervised_candidate_failure_returns_actionable_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_benchmark(_config_dir: Path, _output_dir: Path) -> None:
        raise UnsupervisedCandidateError("tsne", "representation_blobs", "fit")

    monkeypatch.setattr(cli_module, "run_unsupervised_benchmark", fail_benchmark)
    result = runner.invoke(
        app,
        [
            "benchmark-unsupervised",
            "--config-dir",
            str(tmp_path / "configs"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 2
    assert "unsupervised benchmark failed" in result.stderr
    assert "tsne" in result.stderr
    assert "representation_blobs" in result.stderr
    assert "fit" in result.stderr
