"""End-to-end Sprint 2 comparison, report, and CLI publication tests."""

import hashlib
import json
from pathlib import Path

import joblib
import pytest
from typer.testing import CliRunner

from learning_atlas.cli import app
from learning_atlas.supervised.comparison import FittedPipeline
from learning_atlas.workflows.supervised_benchmark import run_supervised_benchmark

pytestmark = pytest.mark.integration


def stable_artifact_hashes(root: Path) -> dict[str, str]:
    """Hash deterministic workflow outputs while excluding volatile manifests."""

    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "benchmark_manifest.json"}
    }


def write_quick_sprint02_configs(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "regression.yaml").write_text(
        "experiment: scratch_regression_benchmark\n"
        "seed: 42\n"
        "n_samples: 120\n"
        "n_features: 6\n"
        "n_informative: 4\n"
        "noise: 5.0\n"
        "cv_folds: 2\n"
        "forest_estimators: 5\n"
        "boosting_estimators: 6\n"
        "max_depth: 3\n",
        encoding="utf-8",
    )
    (root / "classification.yaml").write_text(
        "experiment: scratch_classification_benchmark\n"
        "seed: 42\n"
        "n_samples: 160\n"
        "n_features: 6\n"
        "n_informative: 4\n"
        "class_sep: 1.8\n"
        "label_noise: 0.02\n"
        "cv_folds: 2\n"
        "forest_estimators: 5\n"
        "boosting_estimators: 6\n"
        "max_depth: 3\n",
        encoding="utf-8",
    )


def test_single_cli_command_publishes_both_tasks_and_recruiter_evidence(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    write_quick_sprint02_configs(configs)
    output = tmp_path / "benchmark"

    result = CliRunner().invoke(
        app,
        [
            "benchmark-supervised",
            "--config-dir",
            str(configs),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stderr
    summary = json.loads(result.stdout)
    assert set(summary) == {
        "scratch_regression_benchmark",
        "scratch_classification_benchmark",
    }
    assert summary["scratch_regression_benchmark"]["metrics"]["test_r2"] > 0.80
    assert summary["scratch_classification_benchmark"]["metrics"]["test_macro_f1"] > 0.80
    assert (output / "comparison.csv").stat().st_size > 500
    assert (output / "comparison.md").stat().st_size > 200
    comparison = json.loads((output / "comparison.json").read_text())
    assert len(comparison["scratch_regression_benchmark"]["candidates"]) == 9
    assert len(comparison["scratch_classification_benchmark"]["candidates"]) == 10
    classification_source = comparison["scratch_classification_benchmark"]["source"]
    assert "candidate_parameters" in classification_source["details"]
    assert "seed_streams" in classification_source["details"]

    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert set(manifest["root_reports"]) == {
        "comparison.csv",
        "comparison.json",
        "comparison.md",
    }
    assert set(manifest["artifacts_sha256"]) >= {
        "comparison.csv",
        "comparison.json",
        "comparison.md",
        "experiments/scratch_regression_benchmark/manifest.json",
        "experiments/scratch_regression_benchmark/resolved_config.json",
        "experiments/scratch_classification_benchmark/manifest.json",
        "experiments/scratch_classification_benchmark/resolved_config.json",
    }
    for relative, expected in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected

    for experiment in summary:
        run_dir = output / "experiments" / experiment
        stored = json.loads((run_dir / "result.json").read_text())
        assert len([name for name in stored["artifacts"] if name.endswith("plot")]) == 6
        assert (run_dir / "manifest.json").is_file()
        candidate_records = json.loads(
            (run_dir / stored["artifacts"]["candidate_metrics_json"]).read_text()
        )
        assert all(record["parameters"] is not None for record in candidate_records)
        assert all(
            record["preprocessing"] in {"standard", "nonnegative"} for record in candidate_records
        )
        model = joblib.load(run_dir / stored["artifacts"]["model"])
        assert isinstance(model, FittedPipeline)


def test_benchmark_is_reproducible_and_rejects_incomplete_or_existing_output(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    write_quick_sprint02_configs(configs)
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_supervised_benchmark(configs, first)
    run_supervised_benchmark(configs, second)
    assert json.loads((first / "comparison.json").read_text()) == json.loads(
        (second / "comparison.json").read_text()
    )
    assert stable_artifact_hashes(first) == stable_artifact_hashes(second)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_supervised_benchmark(configs, first)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "regression.yaml").write_text(
        (configs / "regression.yaml").read_text(), encoding="utf-8"
    )
    unpublished = tmp_path / "unpublished"
    with pytest.raises(ValueError, match="requires exactly"):
        run_supervised_benchmark(incomplete, unpublished)
    assert not unpublished.exists()
