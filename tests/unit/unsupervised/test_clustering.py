"""Unsupervised benchmark target-isolation and edge-case tests."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import ClusteringBenchmarkConfig
from learning_atlas.core.contracts import LearningParadigm, RunContext
from learning_atlas.unsupervised.clustering import ClusteringBenchmark, label_free_silhouette

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros(4, dtype=np.int64),
        np.full(4, -1, dtype=np.int64),
        np.array([0, -1, -1, -1], dtype=np.int64),
    ],
)
def test_degenerate_silhouette_has_explicit_sentinel(labels: np.ndarray) -> None:
    features = np.arange(8, dtype=np.float64).reshape(4, 2)
    assert label_free_silhouette(features, labels) == -1.0


def test_label_free_silhouette_scores_separated_clusters() -> None:
    features = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    assert label_free_silhouette(features, labels) > 0.95


def test_clustering_never_uses_targets_for_fit_and_publishes_evidence(tmp_path: Path) -> None:
    config = ClusteringBenchmarkConfig(
        n_samples=200,
        kmeans_n_init=3,
        dbscan_eps=0.3,
        dbscan_min_samples=5,
    )
    result = ClusteringBenchmark(config).run(
        RunContext(seed=config.seed, artifacts=ArtifactStore(tmp_path))
    )

    assert result.paradigm is LearningParadigm.UNSUPERVISED
    assert result.source.target_used_for_fit is False
    assert (
        result.selected_model
        == max(
            result.candidates,
            key=lambda candidate: candidate.metrics["selection_score"],
        ).name
    )
    assert {candidate.name for candidate in result.candidates} == {"kmeans", "dbscan"}
    assert all(candidate.metrics["clusters_found"] >= 1 for candidate in result.candidates)
    assert max(candidate.metrics["adjusted_rand"] for candidate in result.candidates) > 0.85
    assert joblib.load(tmp_path / result.artifacts["model"])
    assignments = np.load(tmp_path / result.artifacts["training_assignments"])
    assert set(assignments.files) == {"kmeans", "dbscan"}
    assert (tmp_path / result.artifacts["cluster_plot"]).stat().st_size > 1_000
    assert (tmp_path / result.artifacts["cluster_metrics_plot"]).stat().st_size > 1_000


def test_fit_is_independent_of_retrospective_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from learning_atlas.unsupervised import clustering as module

    features, labels = module.make_moons(n_samples=200, noise=0.08, random_state=42)

    def run_with_labels(evaluation_labels: np.ndarray, name: str):
        monkeypatch.setattr(
            module,
            "make_moons",
            lambda **_kwargs: (features.copy(), evaluation_labels.copy()),
        )
        config = ClusteringBenchmarkConfig(
            n_samples=200,
            kmeans_n_init=3,
            dbscan_eps=0.3,
            dbscan_min_samples=5,
        )
        return ClusteringBenchmark(config).run(
            RunContext(seed=42, artifacts=ArtifactStore(tmp_path / name))
        )

    original = run_with_labels(labels, "original")
    shuffled = run_with_labels(np.random.default_rng(7).permutation(labels), "shuffled")
    label_free_names = {"silhouette", "clusters_found", "noise_fraction", "selection_score"}

    assert original.selected_model == shuffled.selected_model
    for first, second in zip(original.candidates, shuffled.candidates, strict=True):
        assert first.name == second.name
        assert {name: first.metrics[name] for name in label_free_names} == {
            name: second.metrics[name] for name in label_free_names
        }
    assert [candidate.metrics["adjusted_rand"] for candidate in original.candidates] != [
        candidate.metrics["adjusted_rand"] for candidate in shuffled.candidates
    ]
