"""End-to-end Sprint 3 unsupervised evidence and publication tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

import learning_atlas.workflows.unsupervised_benchmark as workflow_module
from learning_atlas.cli import app
from learning_atlas.workflows.unsupervised_benchmark import run_unsupervised_benchmark

pytestmark = pytest.mark.integration

_EXPECTED_EXPERIMENTS = {
    "scratch_clustering_benchmark",
    "scratch_representation_benchmark",
}
_CLUSTERING_CANDIDATES = {
    "kmeans_plus_plus",
    "gaussian_mixture",
    "dbscan",
    "agglomerative_single",
    "agglomerative_complete",
    "agglomerative_average",
    "agglomerative_ward",
}


@dataclass(frozen=True, slots=True)
class _PublishedSuite:
    configs: Path
    output: Path
    summary: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_artifact_hashes(root: Path) -> dict[str, str]:
    """Hash deterministic outputs while excluding timestamped manifests."""

    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "benchmark_manifest.json"}
    }


def _write_quick_sprint03_configs(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "clustering.yaml").write_text(
        "experiment: scratch_clustering_benchmark\n"
        "seed: 42\n"
        "n_samples: 120\n"
        "stability_trials: 2\n"
        "perturbation_scale: 0.025\n"
        "kmeans_n_init: 2\n"
        "kmeans_max_iter: 60\n"
        "gmm_n_init: 1\n"
        "gmm_max_iter: 60\n"
        "dbscan_eps: 0.24\n"
        "dbscan_min_samples: 6\n",
        encoding="utf-8",
    )
    (root / "representation.yaml").write_text(
        "experiment: scratch_representation_benchmark\n"
        "seed: 42\n"
        "n_samples: 90\n"
        "n_features: 5\n"
        "clusters: 3\n"
        "pca_components: 2\n"
        "tsne_perplexity: 15.0\n"
        "tsne_iterations: 250\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def published_suite(tmp_path_factory: pytest.TempPathFactory) -> _PublishedSuite:
    root = tmp_path_factory.mktemp("sprint03-benchmark")
    configs = root / "configs"
    output = root / "published"
    _write_quick_sprint03_configs(configs)
    completed = CliRunner().invoke(
        app,
        [
            "benchmark-unsupervised",
            "--config-dir",
            str(configs),
            "--output-dir",
            str(output),
        ],
    )
    assert completed.exit_code == 0, completed.stderr
    return _PublishedSuite(configs, output, json.loads(completed.stdout))


def _assert_manifest_integrity(root: Path, manifest_name: str) -> dict[str, object]:
    manifest = cast(
        dict[str, object],
        json.loads((root / manifest_name).read_text(encoding="utf-8")),
    )
    recorded = manifest["artifacts_sha256"]
    assert isinstance(recorded, dict)
    physical = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_name
    }
    assert set(recorded) == physical
    for relative, expected_digest in recorded.items():
        assert _sha256(root / relative) == expected_digest
    return manifest


def test_cli_publishes_complete_label_isolated_recruiter_evidence(
    published_suite: _PublishedSuite,
) -> None:
    output = published_suite.output
    summary = published_suite.summary
    assert set(summary) == _EXPECTED_EXPERIMENTS

    clustering_summary = summary["scratch_clustering_benchmark"]
    representation_summary = summary["scratch_representation_benchmark"]
    assert isinstance(clustering_summary, dict)
    assert isinstance(representation_summary, dict)
    assert clustering_summary["metrics"]["selection_score"] > 0.45
    assert clustering_summary["metrics"]["assigned_coverage_mean"] > 0.70
    assert representation_summary["metrics"]["neighborhood_preservation"] > 0.40
    assert representation_summary["metrics"]["distance_correlation"] > 0.25

    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert {
        candidate["name"] for candidate in comparison["scratch_clustering_benchmark"]["candidates"]
    } == _CLUSTERING_CANDIDATES
    assert {
        candidate["name"]
        for candidate in comparison["scratch_representation_benchmark"]["candidates"]
    } == {"pca", "tsne"}
    assert all(not study["source"]["target_used_for_fit"] for study in comparison.values())
    assert all(
        study["source"]["details"]["labels_reserved_for_retrospective_evaluation"]
        for study in comparison.values()
    )

    for report in ("comparison.csv", "comparison.json", "comparison.md"):
        assert (output / report).stat().st_size > 200
    root_manifest = _assert_manifest_integrity(output, "benchmark_manifest.json")
    assert root_manifest["workflow"] == "sprint-03-unsupervised-benchmark"
    root_reports = root_manifest["root_reports"]
    assert isinstance(root_reports, list)
    assert set(root_reports) == {
        "comparison.csv",
        "comparison.json",
        "comparison.md",
    }

    for experiment in _EXPECTED_EXPERIMENTS:
        run_dir = output / "experiments" / experiment
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        assert result["paradigm"] == "unsupervised"
        assert result["source"]["target_used_for_fit"] is False
        assert result["selected_model"] in {candidate["name"] for candidate in result["candidates"]}
        assert len(result["artifacts"].values()) == len(set(result["artifacts"].values()))
        assert all((run_dir / relative).is_file() for relative in result["artifacts"].values())
        _assert_manifest_integrity(run_dir, "manifest.json")

        records_path = run_dir / result["artifacts"]["candidate_records_json"]
        records = json.loads(records_path.read_text(encoding="utf-8"))
        assert all(isinstance(record["seed"], int) for record in records)
        assert all(record["parameters"] for record in records)
        plots = sorted(run_dir.glob("plots/*.png"))
        minimum_plot_count = 8 if experiment == "scratch_clustering_benchmark" else 5
        assert len(plots) >= minimum_plot_count
        assert all(plot.stat().st_size > 5_000 for plot in plots)

    clustering = comparison["scratch_clustering_benchmark"]
    representation = comparison["scratch_representation_benchmark"]
    assert clustering["metrics"]["retrospective_ari_mean"] > 0.45
    assert representation["metrics"]["retrospective_ari"] > 0.70
    tsne_metrics = next(
        candidate["metrics"]
        for candidate in representation["candidates"]
        if candidate["name"] == "tsne"
    )
    assert tsne_metrics["final_kl_divergence"] > 0.0
    assert tsne_metrics["perplexity_max_error"] < 1e-3
    representation_dir = output / "experiments" / "scratch_representation_benchmark"
    representation_result = json.loads(
        (representation_dir / "result.json").read_text(encoding="utf-8")
    )
    sensitivity_records = json.loads(
        (
            representation_dir / representation_result["artifacts"]["tsne_sensitivity_records"]
        ).read_text(encoding="utf-8")
    )
    perplexity_study = [row for row in sensitivity_records if row["study"] == "perplexity"]
    seed_study = [row for row in sensitivity_records if row["study"] == "seed"]
    assert len(perplexity_study) == len(seed_study) == 3
    assert {row["initialization"] for row in perplexity_study} == {"pca"}
    assert len({row["seed"] for row in perplexity_study}) == 1
    assert {row["initialization"] for row in seed_study} == {"random"}
    assert len({row["seed"] for row in seed_study}) == 3
    assert all(row["converged"] is True for row in sensitivity_records)


def test_benchmark_replay_is_byte_stable_and_refuses_overwrite(
    published_suite: _PublishedSuite,
) -> None:
    replay = published_suite.output.parent / "replay"
    run_unsupervised_benchmark(published_suite.configs, replay)

    assert json.loads((published_suite.output / "comparison.json").read_text()) == json.loads(
        (replay / "comparison.json").read_text()
    )
    assert _stable_artifact_hashes(published_suite.output) == _stable_artifact_hashes(replay)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_unsupervised_benchmark(published_suite.configs, replay)


def test_workflow_failure_removes_partial_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed-publication"

    def fail_after_partial_write(_config_dir: Path, staging: Path) -> None:
        staging.mkdir(parents=True)
        (staging / "partial.txt").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("injected suite failure")

    monkeypatch.setattr(workflow_module, "run_suite", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected suite failure"):
        run_unsupervised_benchmark(tmp_path / "unused-configs", output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed-publication.*"))
