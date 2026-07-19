"""Command-line behavior tests through Typer's public runner."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from learning_atlas.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_discovery_version_schema_and_validation() -> None:
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert version.stdout.strip() == "0.1.0"

    discovered = runner.invoke(app, ["list"])
    assert discovered.exit_code == 0
    assert "regression_benchmark\tsupervised" in discovered.stdout
    assert "classification_benchmark\tsupervised" in discovered.stdout
    assert "q_learning_frozen_lake\treinforcement" in discovered.stdout

    schema = runner.invoke(app, ["schema"])
    assert schema.exit_code == 0
    assert "clustering_benchmark" in schema.stdout
    assert "ClusteringBenchmarkConfig" in json.loads(schema.stdout)["$defs"]

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
