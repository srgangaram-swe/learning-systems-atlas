"""Label-isolated clustering comparison on non-convex synthetic structure."""

import joblib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learning_atlas.core.config import ClusteringBenchmarkConfig
from learning_atlas.core.contracts import (
    CandidateResult,
    LearningParadigm,
    RunContext,
    RunResult,
    SourceKind,
    SourceMetadata,
)
from learning_atlas.core.data import array_fingerprint
from learning_atlas.reporting.plots import publish_figure

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def build_candidates(config: ClusteringBenchmarkConfig) -> dict[str, Pipeline]:
    """Build clustering candidates that own their scaling transformation."""

    return {
        "kmeans": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "cluster",
                    KMeans(
                        n_clusters=config.kmeans_clusters,
                        n_init=config.kmeans_n_init,
                        random_state=config.seed,
                    ),
                ),
            ]
        ),
        "dbscan": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "cluster",
                    DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples),
                ),
            ]
        ),
    }


def label_free_silhouette(features: FloatArray, labels: IntArray) -> float:
    """Score non-noise assignments; return -1 when a silhouette is undefined."""

    assigned = labels != -1
    assigned_labels = labels[assigned]
    cluster_count = len(np.unique(assigned_labels))
    if cluster_count < 2 or assigned.sum() <= cluster_count:
        return -1.0
    return float(silhouette_score(features[assigned], assigned_labels))


class ClusteringBenchmark:
    """Compare centroid and density-based clustering without fitting on labels."""

    def __init__(self, config: ClusteringBenchmarkConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        features_raw, labels_raw = make_moons(
            n_samples=self._config.n_samples,
            noise=self._config.noise,
            random_state=context.seed,
        )
        features = np.asarray(features_raw, dtype=np.float64)
        evaluation_labels = np.asarray(labels_raw, dtype=np.int64)

        pipelines = build_candidates(self._config)
        candidate_results: list[CandidateResult] = []
        assignments: dict[str, IntArray] = {}
        for name, pipeline in pipelines.items():
            predicted = np.asarray(pipeline.fit_predict(features), dtype=np.int64)
            assignments[name] = predicted
            scaled = np.asarray(pipeline.named_steps["scale"].transform(features), dtype=np.float64)
            cluster_labels = set(int(label) for label in predicted if label != -1)
            candidate_results.append(
                CandidateResult(
                    name=name,
                    metrics={
                        "silhouette": label_free_silhouette(scaled, predicted),
                        "adjusted_rand": float(adjusted_rand_score(evaluation_labels, predicted)),
                        "normalized_mutual_info": float(
                            normalized_mutual_info_score(evaluation_labels, predicted)
                        ),
                        "clusters_found": float(len(cluster_labels)),
                        "noise_fraction": float(np.mean(predicted == -1)),
                    },
                )
            )

        candidate_results = [
            candidate.model_copy(
                update={
                    "metrics": {
                        **candidate.metrics,
                        "selection_score": candidate.metrics["silhouette"]
                        * (1.0 - candidate.metrics["noise_fraction"]),
                    }
                }
            )
            for candidate in candidate_results
        ]

        winner = max(candidate_results, key=lambda candidate: candidate.metrics["selection_score"])
        model_path = "models/clustering.joblib"
        with context.artifacts.atomic_target(model_path) as temporary:
            joblib.dump(pipelines[winner.name], temporary)
        assignments_path = "models/training_assignments.npz"
        with context.artifacts.atomic_target(assignments_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    kmeans=assignments["kmeans"],
                    dbscan=assignments["dbscan"],
                )
        comparison_path = self._comparison_plot(
            features,
            assignments,
            evaluation_labels,
            context,
        )
        metrics_path = self._metrics_plot(candidate_results, context)

        return RunResult(
            experiment=self._config.experiment,
            paradigm=LearningParadigm.UNSUPERVISED,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.GENERATOR,
                name="scikit-learn noisy moons generator",
                version="make_moons-v1",
                fingerprint_sha256=array_fingerprint(features, evaluation_labels),
                target_used_for_fit=False,
                details={
                    "sample_count": len(features),
                    "feature_count": features.shape[1],
                    "generator_seed": context.seed,
                    "noise": self._config.noise,
                    "labels_reserved_for_retrospective_evaluation": True,
                    "inference_contract": "transductive clustering; refit for new observations",
                },
            ),
            selected_model=winner.name,
            metrics=winner.metrics,
            candidates=tuple(candidate_results),
            artifacts={
                "model": model_path,
                "training_assignments": assignments_path,
                "cluster_plot": comparison_path,
                "cluster_metrics_plot": metrics_path,
            },
            notes=(
                "Selection uses silhouette times assigned coverage; ARI and NMI are retrospective.",
                "A target array is never passed to either candidate's fit method.",
                "DBSCAN has no out-of-sample predict API; persisted assignments are transductive.",
            ),
        )

    @staticmethod
    def _comparison_plot(
        features: FloatArray,
        assignments: dict[str, IntArray],
        evaluation_labels: IntArray,
        context: RunContext,
    ) -> str:
        panels = {"retrospective truth": evaluation_labels, **assignments}
        figure, axes = plt.subplots(1, len(panels), figsize=(15.0, 4.2), sharex=True, sharey=True)
        for axis, (name, labels) in zip(np.atleast_1d(axes), panels.items(), strict=True):
            axis.scatter(features[:, 0], features[:, 1], c=labels, cmap="viridis", s=14, alpha=0.8)
            axis.set_title(name)
            axis.set_xlabel("feature 1")
            axis.grid(alpha=0.15)
        axes[0].set_ylabel("feature 2")
        figure.suptitle("Structure discovery; truth is shown only for retrospective evaluation")
        return publish_figure(figure, context.artifacts, "plots/clustering_comparison.png")

    @staticmethod
    def _metrics_plot(candidates: list[CandidateResult], context: RunContext) -> str:
        metric_names = ("selection_score", "silhouette", "adjusted_rand")
        labels = ("selection score", "silhouette", "retrospective ARI")
        positions = np.arange(len(candidates))
        width = 0.24
        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        for index, (metric_name, label) in enumerate(zip(metric_names, labels, strict=True)):
            offset = (index - 1) * width
            axis.bar(
                positions + offset,
                [candidate.metrics[metric_name] for candidate in candidates],
                width,
                label=label,
            )
        axis.set(
            title="Internal selection and external recovery tell different stories",
            ylabel="Score",
            ylim=(-0.05, 1.05),
            xticks=positions,
            xticklabels=[candidate.name for candidate in candidates],
        )
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        return publish_figure(figure, context.artifacts, "plots/clustering_metrics.png")
