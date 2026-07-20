"""End-to-end Sprint 4 deep-learning evidence and publication tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

import learning_atlas.cli as cli_module
import learning_atlas.workflows.deep_benchmark as workflow_module
from learning_atlas.cli import app
from learning_atlas.deep.training import DeepTrainingError
from learning_atlas.workflows.deep_benchmark import run_deep_benchmark

pytestmark = pytest.mark.integration

_EXPECTED_EXPERIMENTS = {
    "scratch_mlp_benchmark",
    "deep_vision_benchmark",
    "deep_sequence_benchmark",
    "deep_autoencoder_benchmark",
}


@dataclass(frozen=True, slots=True)
class _PublishedSuite:
    configs: Path
    output: Path
    summary: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "benchmark_manifest.json"}
    }


def _manifest(root: Path, name: str) -> dict[str, object]:
    payload = cast(dict[str, object], json.loads((root / name).read_text(encoding="utf-8")))
    checksums = payload["artifacts_sha256"]
    assert isinstance(checksums, dict)
    physical = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != name
    }
    assert set(checksums) == physical
    for relative, expected in checksums.items():
        assert isinstance(relative, str)
        assert _sha256(root / relative) == expected
    return payload


@pytest.fixture(scope="module")
def published_suite(tmp_path_factory: pytest.TempPathFactory) -> _PublishedSuite:
    output = tmp_path_factory.mktemp("sprint04") / "published"
    configs = Path("configs/deep/sprint-04")
    completed = CliRunner().invoke(
        app,
        [
            "benchmark-deep",
            "--config-dir",
            str(configs),
            "--output-dir",
            str(output),
        ],
    )
    assert completed.exit_code == 0, completed.stderr
    return _PublishedSuite(configs=configs, output=output, summary=json.loads(completed.stdout))


def _candidate(study: dict[str, object], name: str) -> dict[str, float]:
    candidates = study["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        assert isinstance(candidate, dict)
        if candidate["name"] == name:
            metrics = candidate["metrics"]
            assert isinstance(metrics, dict)
            return cast(dict[str, float], metrics)
    raise AssertionError(f"missing candidate {name}")


def test_cli_publishes_complete_validation_isolated_learning_evidence(
    published_suite: _PublishedSuite,
) -> None:
    output = published_suite.output
    assert set(published_suite.summary) == _EXPECTED_EXPERIMENTS
    comparison = cast(
        dict[str, dict[str, object]],
        json.loads((output / "comparison.json").read_text(encoding="utf-8")),
    )
    assert set(comparison) == _EXPECTED_EXPERIMENTS

    scratch = comparison["scratch_mlp_benchmark"]
    scratch_mlp = _candidate(scratch, "autograd_mlp")
    linear = _candidate(scratch, "logistic_baseline")
    assert scratch["selected_model"] == "autograd_mlp"
    assert scratch_mlp["gradient_check_max_abs_error"] < 1e-5
    assert scratch_mlp["mean_validation_accuracy"] >= 0.95
    assert scratch_mlp["mean_test_accuracy"] >= 0.95
    assert scratch_mlp["mean_test_accuracy"] - linear["mean_test_accuracy"] >= 0.30

    vision = comparison["deep_vision_benchmark"]
    cnn = _candidate(vision, "compact_cnn")
    image_mlp = _candidate(vision, "vision_mlp")
    assert vision["selected_model"] == "compact_cnn"
    assert cnn["selection_validation_accuracy"] > image_mlp["selection_validation_accuracy"]
    assert cnn["test_accuracy"] > image_mlp["test_accuracy"]
    assert cnn["test_accuracy"] >= 0.94
    for metric in (
        "checkpoint_reload_state_max_abs_delta",
        "checkpoint_reload_logit_max_abs_delta",
        "checkpoint_reload_loss_delta",
        "checkpoint_reload_accuracy_delta",
    ):
        assert cnn[metric] == 0.0

    sequence = comparison["deep_sequence_benchmark"]
    lstm = _candidate(sequence, "sequence_lstm")
    padded_mlp = _candidate(sequence, "padded_mlp")
    assert sequence["selected_model"] == "sequence_lstm"
    assert lstm["selection_validation_accuracy"] >= 0.95
    assert lstm["test_accuracy"] - padded_mlp["test_accuracy"] >= 0.30
    assert lstm["padding_invariance_max_logit_delta"] <= 1e-7

    autoencoder = comparison["deep_autoencoder_benchmark"]
    ae = _candidate(autoencoder, "bottleneck_autoencoder")
    assert autoencoder["selected_model"] == "bottleneck_autoencoder"
    assert ae["selection_validation_auroc"] >= 0.90
    assert ae["anomaly_auroc"] >= 0.90
    assert ae["latent_class_silhouette"] > 0.10
    assert ae["error_separation_ratio"] > 2.0
    autoencoder_records = json.loads(
        (
            output
            / "experiments"
            / "deep_autoencoder_benchmark"
            / "records"
            / "candidate_records.json"
        ).read_text(encoding="utf-8")
    )
    record_seeds = {record["candidate"]: record["seed"] for record in autoencoder_records}
    assert isinstance(record_seeds["bottleneck_autoencoder"], int)
    assert record_seeds["pca_reconstruction"] is None

    root_manifest = _manifest(output, "benchmark_manifest.json")
    assert root_manifest["workflow"] == "sprint-04-deep-learning-benchmark"
    for report in ("comparison.csv", "comparison.json", "comparison.md"):
        assert (output / report).stat().st_size > 300

    expected_plots = {
        "scratch_mlp_benchmark": {
            "mlp_decision_boundaries.png",
            "mlp_loss_curves.png",
        },
        "deep_vision_benchmark": {
            "vision_confusion_matrix.png",
            "vision_conv_filters.png",
            "vision_misclassified.png",
            "vision_training_loss.png",
            "vision_validation_accuracy.png",
        },
        "deep_sequence_benchmark": {
            "sequence_accuracy_by_length.png",
            "sequence_training_loss.png",
            "sequence_validation_accuracy.png",
        },
        "deep_autoencoder_benchmark": {
            "autoencoder_error_histogram.png",
            "autoencoder_latent.png",
            "autoencoder_reconstructions.png",
            "autoencoder_roc.png",
            "autoencoder_training_loss.png",
        },
    }
    expected_models = {
        "scratch_mlp_benchmark": {"autograd_mlp_parameters.npz"},
        "deep_vision_benchmark": {
            "compact_cnn_checkpoint.pt",
            "vision_mlp_checkpoint.pt",
        },
        "deep_sequence_benchmark": {
            "padded_mlp_state.pt",
            "sequence_lstm_state.pt",
        },
        "deep_autoencoder_benchmark": {"bottleneck_autoencoder_state.pt"},
    }
    for experiment, plot_names in expected_plots.items():
        run_dir = output / "experiments" / experiment
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        assert result["paradigm"] == "deep"
        assert result["selected_model"] in {candidate["name"] for candidate in result["candidates"]}
        assert len(result["artifacts"].values()) == len(set(result["artifacts"].values()))
        assert all((run_dir / relative).is_file() for relative in result["artifacts"].values())
        assert {path.name for path in run_dir.glob("plots/*.png")} == plot_names
        assert {path.name for path in run_dir.glob("models/*")} == expected_models[experiment]
        assert all(path.stat().st_size > 5_000 for path in run_dir.glob("plots/*.png"))
        _manifest(run_dir, "manifest.json")

    assert comparison["deep_autoencoder_benchmark"]["source"]["target_used_for_fit"] is False  # type: ignore[index]


def test_reference_replay_is_byte_stable_and_refuses_overwrite(
    published_suite: _PublishedSuite,
) -> None:
    replay = published_suite.output.parent / "replay"
    run_deep_benchmark(published_suite.configs, replay)

    assert json.loads((published_suite.output / "comparison.json").read_text()) == json.loads(
        (replay / "comparison.json").read_text()
    )
    assert _stable_artifact_hashes(published_suite.output) == _stable_artifact_hashes(replay)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_deep_benchmark(published_suite.configs, replay)


def test_workflow_failure_removes_partial_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed-publication"

    def fail_after_partial_write(_config_dir: Path, staging: Path) -> None:
        staging.mkdir(parents=True)
        (staging / "partial.txt").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("injected deep-suite failure")

    monkeypatch.setattr(workflow_module, "run_suite", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected deep-suite failure"):
        run_deep_benchmark(tmp_path / "unused", output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed-publication.*"))


def test_deep_discovery_schema_and_actionable_cli_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovered = CliRunner().invoke(app, ["list"])
    assert discovered.exit_code == 0
    assert discovered.stdout.count("\tdeep\t") == 4

    schema = CliRunner().invoke(app, ["schema"])
    assert schema.exit_code == 0
    definitions = json.loads(schema.stdout)["$defs"]
    assert {
        "ScratchMLPBenchmarkConfig",
        "DeepVisionBenchmarkConfig",
        "DeepSequenceBenchmarkConfig",
        "DeepAutoencoderBenchmarkConfig",
    }.issubset(definitions)

    def fail(_config_dir: Path, _output_dir: Path) -> None:
        raise DeepTrainingError("injected non-finite gradient")

    monkeypatch.setattr(cli_module, "run_deep_benchmark", fail)
    failed = CliRunner().invoke(
        app,
        [
            "benchmark-deep",
            "--config-dir",
            str(tmp_path / "configs"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    assert failed.exit_code == 2
    assert "deep benchmark failed" in failed.stderr
    assert "non-finite gradient" in failed.stderr
