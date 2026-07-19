"""Scientific-isolation contracts spanning the Sprint 3 comparison layer."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pytest

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import (
    ScratchClusteringBenchmarkConfig,
    ScratchRepresentationBenchmarkConfig,
)
from learning_atlas.core.contracts import RunContext, RunResult
from learning_atlas.supervised.datasets import BlobsDataset, make_blobs
from learning_atlas.unsupervised import comparison as comparison_module
from learning_atlas.unsupervised.comparison import ScratchRepresentationBenchmark
from learning_atlas.unsupervised.datasets import make_isotropic_blobs, make_moons
from learning_atlas.unsupervised.density import DBSCAN
from learning_atlas.unsupervised.evaluation import select_highest
from learning_atlas.unsupervised.hierarchical import AgglomerativeClustering
from learning_atlas.unsupervised.manifold import TSNE

pytestmark = pytest.mark.unit

_LABEL_FREE_CANDIDATE_METRICS = (
    "selection_score",
    "silhouette_mean",
    "assigned_coverage_mean",
    "stability_mean",
    "stability_std",
    "mean_clusters_found",
)
_LABEL_FREE_REPRESENTATION_METRICS = (
    "selection_score",
    "neighborhood_preservation",
    "distance_correlation",
)


def _clustering_config() -> ScratchClusteringBenchmarkConfig:
    return ScratchClusteringBenchmarkConfig(
        seed=73,
        n_samples=120,
        stability_trials=2,
        perturbation_scale=0.02,
        kmeans_n_init=2,
        kmeans_max_iter=120,
        gmm_n_init=1,
        gmm_max_iter=180,
        dbscan_eps=0.24,
        dbscan_min_samples=4,
    )


def _candidate_metrics(result: RunResult) -> dict[str, dict[str, float]]:
    return {candidate.name: candidate.metrics for candidate in result.candidates}


def _json_records(root: Path, result: RunResult, artifact: str) -> list[dict[str, Any]]:
    path = root / result.artifacts[artifact]
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def test_retrospective_labels_and_candidate_order_cannot_perturb_clustering_state() -> None:
    """Truth and registry order must be causally downstream of every fitted state."""

    config = _clustering_config()
    datasets = (
        make_isotropic_blobs(n_samples=42, seed=101),
        make_moons(n_samples=42, noise=0.04, seed=202),
    )
    relabeled = tuple(
        replace(dataset, evaluation_labels=np.roll(dataset.evaluation_labels, shift=7))
        for dataset in datasets
    )
    specs = comparison_module._cluster_specs(config)

    baseline = comparison_module._fit_clustering_datasets(config, datasets, specs)
    changed_truth = comparison_module._fit_clustering_datasets(config, relabeled, specs)
    reversed_order = comparison_module._fit_clustering_datasets(
        config,
        datasets,
        tuple(reversed(specs)),
    )

    retrospective_changed = False
    for baseline_run, relabeled_run, reversed_run in zip(
        baseline,
        changed_truth,
        reversed_order,
        strict=True,
    ):
        assert baseline_run.seeds == relabeled_run.seeds == reversed_run.seeds
        np.testing.assert_array_equal(baseline_run.features, relabeled_run.features)
        np.testing.assert_array_equal(baseline_run.features, reversed_run.features)
        for spec in specs:
            name = spec.name
            np.testing.assert_array_equal(
                baseline_run.assignments[name],
                relabeled_run.assignments[name],
            )
            np.testing.assert_array_equal(
                baseline_run.assignments[name],
                reversed_run.assignments[name],
            )
            assert pickle.dumps(baseline_run.models[name], protocol=5) == pickle.dumps(
                relabeled_run.models[name], protocol=5
            )
            assert pickle.dumps(baseline_run.models[name], protocol=5) == pickle.dumps(
                reversed_run.models[name], protocol=5
            )
            assert baseline_run.metrics[name].keys() == relabeled_run.metrics[name].keys()
            for metric in baseline_run.metrics[name]:
                if metric.startswith("retrospective_"):
                    continue
                assert baseline_run.metrics[name][metric] == relabeled_run.metrics[name][metric]
                assert baseline_run.metrics[name][metric] == reversed_run.metrics[name][metric]
            retrospective_changed |= (
                baseline_run.metrics[name]["retrospective_ari"]
                != relabeled_run.metrics[name]["retrospective_ari"]
            )
    assert retrospective_changed, "the adversarial relabeling must exercise the test oracle"

    baseline_stability = comparison_module._stability_trials(config, baseline, specs)
    relabeled_stability = comparison_module._stability_trials(config, changed_truth, specs)
    reversed_stability = comparison_module._stability_trials(
        config,
        reversed_order,
        tuple(reversed(specs)),
    )
    assert baseline_stability == relabeled_stability == reversed_stability

    baseline_candidates = {
        candidate.name: candidate.metrics
        for candidate in comparison_module._aggregate_cluster_candidates(
            baseline,
            specs,
            baseline_stability,
        )
    }
    relabeled_candidates = {
        candidate.name: candidate.metrics
        for candidate in comparison_module._aggregate_cluster_candidates(
            changed_truth,
            specs,
            relabeled_stability,
        )
    }
    reversed_candidates = {
        candidate.name: candidate.metrics
        for candidate in comparison_module._aggregate_cluster_candidates(
            reversed_order,
            tuple(reversed(specs)),
            reversed_stability,
        )
    }
    assert baseline_candidates.keys() == relabeled_candidates.keys() == reversed_candidates.keys()
    for name in baseline_candidates:
        for metric in _LABEL_FREE_CANDIDATE_METRICS:
            assert baseline_candidates[name][metric] == relabeled_candidates[name][metric]
            assert baseline_candidates[name][metric] == reversed_candidates[name][metric]

    baseline_winner = select_highest(
        {name: metrics["selection_score"] for name, metrics in baseline_candidates.items()}
    )
    relabeled_winner = select_highest(
        {name: metrics["selection_score"] for name, metrics in relabeled_candidates.items()}
    )
    reversed_winner = select_highest(
        {name: metrics["selection_score"] for name, metrics in reversed_candidates.items()}
    )
    assert baseline_winner == relabeled_winner == reversed_winner


def test_representation_benchmark_is_label_isolated_and_persists_tsne_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing truth may alter retrospective evidence, never representation selection."""

    config = ScratchRepresentationBenchmarkConfig(
        seed=73,
        n_samples=90,
        n_features=5,
        clusters=3,
        pca_components=2,
        tsne_perplexity=12.0,
        tsne_iterations=250,
    )
    baseline_root = tmp_path / "baseline"
    baseline = ScratchRepresentationBenchmark(config).run(
        RunContext(seed=config.seed, artifacts=ArtifactStore(baseline_root))
    )

    original_make_blobs = make_blobs

    def relabeled_make_blobs(**kwargs: Any) -> BlobsDataset:
        generated = original_make_blobs(**kwargs)
        targets = np.roll(generated.targets, shift=11).copy()
        targets.setflags(write=False)
        return BlobsDataset(
            features=generated.features,
            targets=targets,
            centers=generated.centers,
            cluster_std=generated.cluster_std,
        )

    monkeypatch.setattr(comparison_module, "make_blobs", relabeled_make_blobs)
    relabeled_root = tmp_path / "relabeled"
    changed_truth = ScratchRepresentationBenchmark(config).run(
        RunContext(seed=config.seed, artifacts=ArtifactStore(relabeled_root))
    )

    assert baseline.selected_model == changed_truth.selected_model
    assert baseline.source.target_used_for_fit is False
    assert changed_truth.source.target_used_for_fit is False
    assert baseline.source.details == changed_truth.source.details
    assert baseline.source.fingerprint_sha256 != changed_truth.source.fingerprint_sha256

    baseline_metrics = _candidate_metrics(baseline)
    changed_metrics = _candidate_metrics(changed_truth)
    assert baseline_metrics.keys() == changed_metrics.keys() == {"pca", "tsne"}
    for candidate in baseline_metrics:
        label_free_metrics = list(_LABEL_FREE_REPRESENTATION_METRICS)
        label_free_metrics.append(
            "explained_variance_ratio" if candidate == "pca" else "final_kl_divergence"
        )
        if candidate == "tsne":
            label_free_metrics.append("perplexity_max_error")
        for metric in label_free_metrics:
            assert baseline_metrics[candidate][metric] == changed_metrics[candidate][metric]
    assert any(
        baseline_metrics[name]["retrospective_ari"] != changed_metrics[name]["retrospective_ari"]
        for name in baseline_metrics
    )

    baseline_records = {
        record["candidate"]: record
        for record in _json_records(
            baseline_root,
            baseline,
            "candidate_records_json",
        )
    }
    changed_records = {
        record["candidate"]: record
        for record in _json_records(
            relabeled_root,
            changed_truth,
            "candidate_records_json",
        )
    }
    for name in baseline_records:
        assert isinstance(baseline_records[name]["seed"], int)
        assert baseline_records[name]["seed"] == changed_records[name]["seed"]
        assert baseline_records[name]["parameters"] == changed_records[name]["parameters"]

    baseline_sensitivity = _json_records(
        baseline_root,
        baseline,
        "tsne_sensitivity_records",
    )
    changed_sensitivity = _json_records(
        relabeled_root,
        changed_truth,
        "tsne_sensitivity_records",
    )
    for baseline_row, changed_row in zip(
        baseline_sensitivity,
        changed_sensitivity,
        strict=True,
    ):
        assert isinstance(baseline_row["seed"], int)
        for key in ("perplexity", "seed", "neighborhood_preservation", "final_kl"):
            assert baseline_row[key] == changed_row[key]

    baseline_models = joblib.load(baseline_root / baseline.artifacts["models"])
    changed_models = joblib.load(relabeled_root / changed_truth.artifacts["models"])
    assert pickle.dumps(baseline_models, protocol=5) == pickle.dumps(
        changed_models,
        protocol=5,
    )
    tsne = baseline_models["tsne"]
    assert isinstance(tsne, TSNE)
    assert tsne.converged_ is True
    assert tsne.kl_divergence_ == tsne.kl_history_[-1]
    assert tsne.kl_divergence_ < tsne.post_exaggeration_start_kl_
    assert (
        np.max(np.abs(np.log(tsne.perplexities_) - np.log(config.tsne_perplexity)))
        <= tsne.perplexity_tolerance + 1e-12
    )
    assert baseline_metrics["tsne"]["perplexity_max_error"] == pytest.approx(
        float(np.max(np.abs(tsne.perplexities_ - config.tsne_perplexity))),
        rel=0.0,
        abs=0.0,
    )

    for artifact in ("models", "pca_diagnostics", "tsne_optimization"):
        assert (baseline_root / baseline.artifacts[artifact]).read_bytes() == (
            relabeled_root / changed_truth.artifacts[artifact]
        ).read_bytes()


def test_failed_refits_preserve_transductive_estimator_state() -> None:
    """Validation failures cannot partially replace an already fitted transductive model."""

    features = np.asarray(
        [[-2.0, 0.0], [-1.8, 0.1], [0.0, 2.0], [0.2, 2.1], [2.0, 0.0], [2.2, 0.1]],
        dtype=np.float64,
    )
    estimators = (
        DBSCAN(eps=0.35, min_samples=2).fit(features),
        AgglomerativeClustering(n_clusters=3, linkage="average").fit(features),
        TSNE(
            perplexity=2.0,
            learning_rate=10.0,
            max_iter=50,
            early_exaggeration_iter=15,
            init="random",
            seed=8,
        ).fit(features),
    )

    for estimator in estimators:
        before = pickle.dumps(estimator, protocol=5)
        with pytest.raises(ValueError, match="finite"):
            estimator.fit([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0]])
        assert estimator.n_features_in_ == 2
        assert pickle.dumps(estimator, protocol=5) == before
