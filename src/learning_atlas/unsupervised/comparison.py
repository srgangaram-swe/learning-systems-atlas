"""Label-isolated, from-scratch unsupervised comparison experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import joblib
import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from learning_atlas.core.config import (
    ScratchClusteringBenchmarkConfig,
    ScratchRepresentationBenchmarkConfig,
)
from learning_atlas.core.contracts import (
    CandidateResult,
    LearningParadigm,
    RunContext,
    RunResult,
    SourceKind,
    SourceMetadata,
)
from learning_atlas.core.data import array_fingerprint
from learning_atlas.core.reproducibility import derive_named_seed
from learning_atlas.core.validation import FloatArray
from learning_atlas.reporting.unsupervised import (
    assignment_grid,
    dbscan_sensitivity_plot,
    dendrogram_plot,
    embedding_comparison_plot,
    hierarchy_diagnostics_plot,
    k_distance_plot,
    mixture_density_plot,
    optimization_histories_plot,
    pca_diagnostics_plot,
    random_partition_ari_plot,
    representation_metrics_plot,
    restart_objectives_plot,
    score_heatmap,
    stability_plot,
    tsne_affinity_diagnostics_plot,
    tsne_sensitivity_plot,
)
from learning_atlas.supervised.datasets import make_blobs
from learning_atlas.supervised.preprocessing import StandardScaler
from learning_atlas.unsupervised.datasets import StructureDataset, clustering_suite
from learning_atlas.unsupervised.decomposition import PCA
from learning_atlas.unsupervised.density import DBSCAN
from learning_atlas.unsupervised.evaluation import (
    clustering_selection_score,
    distance_correlation,
    neighborhood_preservation,
    select_highest,
)
from learning_atlas.unsupervised.hierarchical import AgglomerativeClustering, Linkage
from learning_atlas.unsupervised.kmeans import KMeans
from learning_atlas.unsupervised.manifold import TSNE
from learning_atlas.unsupervised.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_analysis,
)
from learning_atlas.unsupervised.mixture import GaussianMixture

IntArray = NDArray[np.int64]
ClusterModel: TypeAlias = KMeans | GaussianMixture | DBSCAN | AgglomerativeClustering
StabilityEvidence: TypeAlias = Mapping[str, Mapping[str, Sequence[float]]]


class UnsupervisedCandidateError(RuntimeError):
    """Attach candidate, dataset, and phase context to numerical failures."""

    def __init__(self, candidate: str, dataset: str, phase: str) -> None:
        self.candidate = candidate
        self.dataset = dataset
        self.phase = phase
        super().__init__(
            f"unsupervised candidate {candidate!r} failed during {phase} on {dataset!r}"
        )


@dataclass(frozen=True, slots=True)
class _ClusterSpec:
    name: str
    factory: Callable[[int, int], ClusterModel]
    parameters: Callable[[int], dict[str, JsonValue]]


@dataclass(slots=True)
class _ClusterDatasetRun:
    dataset: StructureDataset
    features: FloatArray
    scaler: StandardScaler
    models: dict[str, ClusterModel]
    assignments: dict[str, IntArray]
    metrics: dict[str, dict[str, float]]
    seeds: dict[str, int]


@dataclass(frozen=True, slots=True)
class _ClusterRecord:
    dataset: str
    candidate: str
    seed: int
    parameters: dict[str, JsonValue]
    metrics: dict[str, float]


def _declared_clusters(dataset: StructureDataset) -> int:
    value = dataset.details.get("clusters")
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"dataset {dataset.name!r} must declare an integer cluster hypothesis"
        raise TypeError(msg)
    return value


def _agglomerative_spec(linkage: Linkage) -> _ClusterSpec:
    def factory(clusters: int, _seed: int) -> ClusterModel:
        return AgglomerativeClustering(clusters, linkage=linkage)

    def parameters(clusters: int) -> dict[str, JsonValue]:
        return {"n_clusters": clusters, "linkage": linkage, "metric": "euclidean"}

    return _ClusterSpec(
        name=f"agglomerative_{linkage}",
        factory=factory,
        parameters=parameters,
    )


def _cluster_specs(config: ScratchClusteringBenchmarkConfig) -> tuple[_ClusterSpec, ...]:
    return (
        _ClusterSpec(
            name="kmeans_plus_plus",
            factory=lambda clusters, seed: KMeans(
                clusters,
                init="k-means++",
                n_init=config.kmeans_n_init,
                max_iter=config.kmeans_max_iter,
                random_state=seed,
            ),
            parameters=lambda clusters: {
                "n_clusters": clusters,
                "init": "k-means++",
                "n_init": config.kmeans_n_init,
                "max_iter": config.kmeans_max_iter,
                "tol": 1e-4,
            },
        ),
        _ClusterSpec(
            name="gaussian_mixture",
            factory=lambda clusters, seed: GaussianMixture(
                clusters,
                init="kmeans",
                n_init=config.gmm_n_init,
                max_iter=config.gmm_max_iter,
                tol=1e-3,
                reg_covar=1e-6,
                random_state=seed,
            ),
            parameters=lambda clusters: {
                "n_components": clusters,
                "init": "kmeans",
                "n_init": config.gmm_n_init,
                "max_iter": config.gmm_max_iter,
                "tol": 1e-3,
                "reg_covar": 1e-6,
            },
        ),
        _ClusterSpec(
            name="dbscan",
            factory=lambda _clusters, _seed: DBSCAN(
                eps=config.dbscan_eps,
                min_samples=config.dbscan_min_samples,
            ),
            parameters=lambda _clusters: {
                "eps": config.dbscan_eps,
                "min_samples": config.dbscan_min_samples,
                "metric": "euclidean",
            },
        ),
        _agglomerative_spec("single"),
        _agglomerative_spec("complete"),
        _agglomerative_spec("average"),
        _agglomerative_spec("ward"),
    )


def _fit_cluster(
    spec: _ClusterSpec,
    features: FloatArray,
    *,
    clusters: int,
    seed: int,
    dataset: str,
    phase: str,
) -> tuple[ClusterModel, IntArray]:
    try:
        model = spec.factory(clusters, seed)
        assignments = np.asarray(model.fit_predict(features), dtype=np.int64)
    except (MemoryError, RuntimeError, TypeError, ValueError) as error:
        raise UnsupervisedCandidateError(spec.name, dataset, phase) from error
    return model, assignments


def _fit_clustering_datasets(
    config: ScratchClusteringBenchmarkConfig,
    datasets: Sequence[StructureDataset],
    specs: Sequence[_ClusterSpec],
) -> list[_ClusterDatasetRun]:
    runs: list[_ClusterDatasetRun] = []
    for dataset in datasets:
        scaler = StandardScaler()
        features = scaler.fit_transform(dataset.features)
        clusters = _declared_clusters(dataset)
        models: dict[str, ClusterModel] = {}
        assignments: dict[str, IntArray] = {}
        metrics: dict[str, dict[str, float]] = {}
        seeds: dict[str, int] = {}
        for spec in specs:
            seed = derive_named_seed(
                config.seed,
                config.experiment,
                "candidate",
                spec.name,
                dataset.name,
                "fit",
            )
            model, predicted = _fit_cluster(
                spec,
                features,
                clusters=clusters,
                seed=seed,
                dataset=dataset.name,
                phase="fit",
            )
            evidence = silhouette_analysis(features, predicted)
            models[spec.name] = model
            assignments[spec.name] = predicted
            seeds[spec.name] = seed
            metrics[spec.name] = {
                "silhouette": evidence.score,
                "silhouette_defined": float(evidence.defined),
                "assigned_coverage": evidence.coverage,
                "clusters_found": float(evidence.n_clusters),
                "noise_fraction": float(np.mean(predicted == -1)),
                "retrospective_ari": adjusted_rand_score(dataset.evaluation_labels, predicted),
                "retrospective_nmi": normalized_mutual_info_score(
                    dataset.evaluation_labels, predicted
                ),
            }
            if isinstance(model, KMeans):
                metrics[spec.name].update(
                    {
                        "inertia": model.inertia_,
                        "iterations": float(model.n_iter_),
                        "converged": float(model.converged_),
                    }
                )
            elif isinstance(model, GaussianMixture):
                metrics[spec.name].update(
                    {
                        "aic": model.aic(features),
                        "bic": model.bic(features),
                        "lower_bound": model.lower_bound_,
                        "parameters": float(model.n_parameters_),
                        "iterations": float(model.n_iter_),
                        "converged": float(model.converged_),
                    }
                )
            elif isinstance(model, DBSCAN):
                roles = model.sample_roles_
                metrics[spec.name].update(
                    {
                        "core_fraction": float(np.mean(roles == "core")),
                        "border_fraction": float(np.mean(roles == "border")),
                    }
                )
            else:
                assert isinstance(model, AgglomerativeClustering)
                metrics[spec.name]["maximum_merge_distance"] = float(model.distances_[-1])
        runs.append(
            _ClusterDatasetRun(
                dataset=dataset,
                features=features,
                scaler=scaler,
                models=models,
                assignments=assignments,
                metrics=metrics,
                seeds=seeds,
            )
        )
    return runs


def _stability_trials(
    config: ScratchClusteringBenchmarkConfig,
    runs: Sequence[_ClusterDatasetRun],
    specs: Sequence[_ClusterSpec],
) -> dict[str, dict[str, tuple[float, ...]]]:
    stability: dict[str, dict[str, list[float]]] = {
        run.dataset.name: {spec.name: [] for spec in specs} for run in runs
    }
    for run in runs:
        for trial in range(config.stability_trials):
            perturbation_seed = derive_named_seed(
                config.seed,
                config.experiment,
                "stability",
                run.dataset.name,
                f"trial-{trial}",
            )
            perturbation = np.random.default_rng(perturbation_seed).normal(
                scale=config.perturbation_scale,
                size=run.features.shape,
            )
            perturbed = np.asarray(run.features + perturbation, dtype=np.float64)
            clusters = _declared_clusters(run.dataset)
            for spec in specs:
                _, predicted = _fit_cluster(
                    spec,
                    perturbed,
                    clusters=clusters,
                    seed=run.seeds[spec.name],
                    dataset=run.dataset.name,
                    phase=f"stability trial {trial}",
                )
                stability[run.dataset.name][spec.name].append(
                    adjusted_rand_score(run.assignments[spec.name], predicted)
                )
    return {
        dataset: {candidate: tuple(values) for candidate, values in candidates.items()}
        for dataset, candidates in stability.items()
    }


def _aggregate_cluster_candidates(
    runs: Sequence[_ClusterDatasetRun],
    specs: Sequence[_ClusterSpec],
    stability: StabilityEvidence,
) -> tuple[CandidateResult, ...]:
    candidates: list[CandidateResult] = []
    for spec in specs:
        records = [run.metrics[spec.name] for run in runs]
        stability_values = tuple(
            value for run in runs for value in stability[run.dataset.name][spec.name]
        )
        stability_mean = float(np.mean(stability_values))
        per_dataset_selection = [
            clustering_selection_score(
                silhouette=record["silhouette"],
                coverage=record["assigned_coverage"],
                stability=float(np.mean(stability[run.dataset.name][spec.name])),
            )
            for run, record in zip(runs, records, strict=True)
        ]
        metrics = {
            "selection_score": float(np.mean(per_dataset_selection)),
            "silhouette_mean": float(np.mean([record["silhouette"] for record in records])),
            "assigned_coverage_mean": float(
                np.mean([record["assigned_coverage"] for record in records])
            ),
            "stability_mean": stability_mean,
            "stability_std": float(np.std(stability_values)),
            "retrospective_ari_mean": float(
                np.mean([record["retrospective_ari"] for record in records])
            ),
            "retrospective_nmi_mean": float(
                np.mean([record["retrospective_nmi"] for record in records])
            ),
            "mean_clusters_found": float(np.mean([record["clusters_found"] for record in records])),
        }
        candidates.append(CandidateResult(name=spec.name, metrics=metrics))
    return tuple(candidates)


def _write_clustering_records(
    context: RunContext,
    runs: Sequence[_ClusterDatasetRun],
    specs: Sequence[_ClusterSpec],
    stability: StabilityEvidence,
) -> tuple[str, str, str, str]:
    records = [
        _ClusterRecord(
            dataset=run.dataset.name,
            candidate=spec.name,
            seed=run.seeds[spec.name],
            parameters=spec.parameters(_declared_clusters(run.dataset)),
            metrics=run.metrics[spec.name],
        )
        for run in runs
        for spec in specs
    ]
    records_path = "metrics/clustering_records.json"
    context.artifacts.write_json(
        records_path,
        [
            {
                "dataset": record.dataset,
                "candidate": record.candidate,
                "seed": record.seed,
                "parameters": record.parameters,
                "metrics": record.metrics,
            }
            for record in records
        ],
    )
    csv_path = "metrics/clustering_records.csv"
    metric_names = sorted({name for record in records for name in record.metrics})
    with context.artifacts.atomic_target(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["dataset", "candidate", "seed", "parameters_json", *metric_names],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "dataset": record.dataset,
                        "candidate": record.candidate,
                        "seed": record.seed,
                        "parameters_json": json.dumps(record.parameters, sort_keys=True),
                        **record.metrics,
                    }
                )

    stability_path = "metrics/stability_trials.json"
    context.artifacts.write_json(
        stability_path,
        {
            dataset: {candidate: list(values) for candidate, values in candidates.items()}
            for dataset, candidates in stability.items()
        },
    )
    convergence: dict[str, JsonValue] = {}
    for run in runs:
        kmeans = run.models["kmeans_plus_plus"]
        mixture = run.models["gaussian_mixture"]
        assert isinstance(kmeans, KMeans)
        assert isinstance(mixture, GaussianMixture)
        convergence[run.dataset.name] = {
            "kmeans": {
                "winning_history": list(kmeans.inertia_history_),
                "restart_histories": [list(history) for history in kmeans.restart_histories_],
                "restart_final_inertia": list(kmeans.restart_inertias_),
            },
            "gaussian_mixture": {
                "winning_log_likelihood": list(mixture.log_likelihood_history_),
                "restart_histories": [list(history) for history in mixture.restart_histories_],
                "restart_lower_bounds": list(mixture.restart_lower_bounds_),
            },
        }
    convergence_path = "metrics/convergence.json"
    context.artifacts.write_json(convergence_path, convergence)
    return records_path, csv_path, stability_path, convergence_path


def _write_assignment_archive(
    context: RunContext,
    runs: Sequence[_ClusterDatasetRun],
    specs: Sequence[_ClusterSpec],
) -> str:
    path = "models/clustering_assignments.npz"
    arrays = {
        f"{run.dataset.name}__{spec.name}": run.assignments[spec.name]
        for run in runs
        for spec in specs
    }
    with context.artifacts.atomic_target(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    return path


def _write_clustering_diagnostics(
    context: RunContext,
    runs: Sequence[_ClusterDatasetRun],
    specs: Sequence[_ClusterSpec],
) -> str:
    """Persist inspectable learned state for every clustering family."""

    arrays: dict[
        str,
        NDArray[np.float64] | NDArray[np.int64] | NDArray[np.str_],
    ] = {}
    for run in runs:
        prefix = run.dataset.name
        arrays[f"{prefix}__standardized_features"] = run.features
        arrays[f"{prefix}__scaler_mean"] = run.scaler.mean_
        arrays[f"{prefix}__scaler_scale"] = run.scaler.scale_
        for spec in specs:
            arrays[f"{prefix}__{spec.name}__assignments"] = run.assignments[spec.name]

        kmeans = run.models["kmeans_plus_plus"]
        mixture = run.models["gaussian_mixture"]
        density = run.models["dbscan"]
        assert isinstance(kmeans, KMeans)
        assert isinstance(mixture, GaussianMixture)
        assert isinstance(density, DBSCAN)
        arrays[f"{prefix}__kmeans__centers"] = kmeans.cluster_centers_
        arrays[f"{prefix}__kmeans__winning_inertia"] = np.asarray(
            kmeans.inertia_history_, dtype=np.float64
        )
        arrays[f"{prefix}__kmeans__restart_inertia"] = np.asarray(
            kmeans.restart_inertias_, dtype=np.float64
        )
        arrays[f"{prefix}__gmm__weights"] = mixture.weights_
        arrays[f"{prefix}__gmm__means"] = mixture.means_
        arrays[f"{prefix}__gmm__covariances"] = mixture.covariances_
        arrays[f"{prefix}__gmm__responsibilities"] = mixture.responsibilities_
        arrays[f"{prefix}__dbscan__core_indices"] = density.core_sample_indices_
        arrays[f"{prefix}__dbscan__neighborhood_counts"] = density.neighborhood_counts_
        arrays[f"{prefix}__dbscan__k_distances"] = density.k_distances_
        arrays[f"{prefix}__dbscan__sample_roles"] = density.sample_roles_
        for linkage in ("single", "complete", "average", "ward"):
            hierarchy = run.models[f"agglomerative_{linkage}"]
            assert isinstance(hierarchy, AgglomerativeClustering)
            arrays[f"{prefix}__agglomerative_{linkage}__linkage"] = hierarchy.linkage_matrix_
            for cluster_count in range(1, min(6, len(run.features)) + 1):
                arrays[f"{prefix}__agglomerative_{linkage}__cut_{cluster_count}"] = hierarchy.cut(
                    cluster_count
                )

    path = "models/clustering_diagnostics.npz"
    with context.artifacts.atomic_target(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    return path


def _metric_diagnostics(
    config: ScratchClusteringBenchmarkConfig,
    context: RunContext,
) -> tuple[str, str]:
    """Publish deterministic chance-baseline and exact-fixture metric evidence."""

    balanced = np.tile(np.arange(4, dtype=np.int64), 60)
    scores: list[float] = []
    seeds: list[int] = []
    for trial in range(96):
        seed = derive_named_seed(
            config.seed,
            config.experiment,
            "metric-diagnostic",
            f"trial-{trial}",
        )
        seeds.append(seed)
        permuted = np.random.default_rng(seed).permutation(balanced)
        scores.append(adjusted_rand_score(balanced, permuted))

    perfect_permutation = np.asarray([2, 2, 0, 0, 1, 1], dtype=np.int64)
    perfect_reference = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    json_seeds: list[JsonValue] = []
    json_scores: list[JsonValue] = []
    json_seeds.extend(seeds)
    json_scores.extend(scores)
    payload: dict[str, JsonValue] = {
        "definitions": {
            "silhouette": "(nearest-other-cluster distance - within-cluster distance) / max of both distances",
            "adjusted_rand": "pair-count agreement adjusted for the generalized hypergeometric chance baseline",
            "normalized_mutual_information": "mutual information normalized by the arithmetic mean of partition entropies",
        },
        "perfect_label_permutation_fixture": {
            "adjusted_rand": adjusted_rand_score(perfect_reference, perfect_permutation),
            "normalized_mutual_information": normalized_mutual_info_score(
                perfect_reference, perfect_permutation
            ),
        },
        "random_partition_study": {
            "trials": len(scores),
            "seeds": json_seeds,
            "scores": json_scores,
            "mean": float(np.mean(scores)),
            "standard_deviation": float(np.std(scores)),
            "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)),
        },
    }
    diagnostics_path = "metrics/metric_diagnostics.json"
    context.artifacts.write_json(diagnostics_path, payload)
    context_label = (
        f"root seed={config.seed}; 96 named random-label trials; "
        "external metrics excluded from model selection"
    )
    plot_path = random_partition_ari_plot(
        scores,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/random_partition_ari.png",
    )
    return diagnostics_path, plot_path


def _clustering_plots(
    config: ScratchClusteringBenchmarkConfig,
    context: RunContext,
    runs: Sequence[_ClusterDatasetRun],
    specs: Sequence[_ClusterSpec],
    stability: StabilityEvidence,
) -> dict[str, str]:
    dataset_names = tuple(run.dataset.name for run in runs)
    candidate_names = tuple(spec.name for spec in specs)
    feature_map = {run.dataset.name: run.features for run in runs}
    assignment_map = {
        (run.dataset.name, spec.name): run.assignments[spec.name] for run in runs for spec in specs
    }
    context_label = (
        f"root seed={config.seed}; feature-only fit/selection; "
        f"n={config.n_samples} per dataset; stability trials={config.stability_trials}"
    )
    artifacts = {
        "assignment_grid": assignment_grid(
            feature_map,
            assignment_map,
            candidate_names,
            context=context_label,
            artifacts=context.artifacts,
            relative_path="plots/clustering_assignments.png",
        )
    }
    matrices = {
        "internal_silhouette": np.asarray(
            [[run.metrics[name]["silhouette"] for run in runs] for name in candidate_names]
        ),
        "assigned_coverage": np.asarray(
            [[run.metrics[name]["assigned_coverage"] for run in runs] for name in candidate_names]
        ),
        "retrospective_ari": np.asarray(
            [[run.metrics[name]["retrospective_ari"] for run in runs] for name in candidate_names]
        ),
    }
    for name, matrix in matrices.items():
        title = {
            "internal_silhouette": "Internal geometry varies by structure and method",
            "assigned_coverage": "Assigned coverage exposes density-method abstention",
            "retrospective_ari": "Retrospective recovery (never used for selection)",
        }[name]
        artifacts[name] = score_heatmap(
            matrix,
            candidate_names,
            dataset_names,
            title=title,
            colorbar_label=name.replace("_", " "),
            context=context_label,
            artifacts=context.artifacts,
            relative_path=f"plots/{name}.png",
            value_range=(-1.0, 1.0) if name != "assigned_coverage" else (0.0, 1.0),
        )
    combined_stability = {
        candidate: tuple(
            value for dataset in dataset_names for value in stability[dataset][candidate]
        )
        for candidate in candidate_names
    }
    artifacts["stability"] = stability_plot(
        combined_stability,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/clustering_stability.png",
    )

    kmeans_histories: dict[str, Sequence[float]] = {}
    mixture_histories: dict[str, Sequence[float]] = {}
    kmeans_restarts: dict[str, Sequence[float]] = {}
    mixture_restarts: dict[str, Sequence[float]] = {}
    for run in runs:
        kmeans = run.models["kmeans_plus_plus"]
        mixture = run.models["gaussian_mixture"]
        assert isinstance(kmeans, KMeans)
        assert isinstance(mixture, GaussianMixture)
        kmeans_histories[run.dataset.name] = kmeans.inertia_history_
        mixture_histories[run.dataset.name] = mixture.lower_bound_history_
        kmeans_restarts[run.dataset.name] = kmeans.restart_inertias_
        mixture_restarts[run.dataset.name] = mixture.restart_lower_bounds_
    artifacts["kmeans_optimization"] = optimization_histories_plot(
        kmeans_histories,
        title="K-means winning-restart inertia",
        ylabel="Within-cluster sum of squares",
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/kmeans_optimization.png",
    )
    artifacts["gmm_optimization"] = optimization_histories_plot(
        mixture_histories,
        title="Gaussian-mixture EM lower bound",
        ylabel="Mean observed-data log likelihood",
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/gmm_optimization.png",
    )
    artifacts["restart_objectives"] = restart_objectives_plot(
        kmeans_restarts,
        mixture_restarts,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/restart_objectives.png",
    )

    anisotropic = next(run for run in runs if run.dataset.name == "anisotropic_blobs")
    mixture = anisotropic.models["gaussian_mixture"]
    assert isinstance(mixture, GaussianMixture)
    artifacts["gmm_density"] = mixture_density_plot(
        anisotropic.features,
        anisotropic.assignments["gaussian_mixture"],
        mixture.means_,
        mixture.covariances_,
        mixture.weights_,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/gmm_density.png",
    )

    moon_run = next(run for run in runs if run.dataset.name == "two_moons")
    eps_values = np.asarray(config.dbscan_eps * np.linspace(0.70, 1.30, 7), dtype=np.float64)
    silhouettes: list[float] = []
    noise_fractions: list[float] = []
    cluster_counts: list[int] = []
    for eps in eps_values:
        model = DBSCAN(eps=float(eps), min_samples=config.dbscan_min_samples)
        predicted = model.fit_predict(moon_run.features)
        evidence = silhouette_analysis(moon_run.features, predicted)
        silhouettes.append(evidence.score)
        noise_fractions.append(float(np.mean(predicted == -1)))
        cluster_counts.append(evidence.n_clusters)
    artifacts["dbscan_sensitivity"] = dbscan_sensitivity_plot(
        eps_values.tolist(),
        silhouettes,
        noise_fractions,
        cluster_counts,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/dbscan_sensitivity.png",
    )
    density_model = moon_run.models["dbscan"]
    assert isinstance(density_model, DBSCAN)
    artifacts["dbscan_k_distance"] = k_distance_plot(
        density_model.k_distances_.tolist(),
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/dbscan_k_distance.png",
    )

    isotropic = next(run for run in runs if run.dataset.name == "isotropic_blobs")
    hierarchy = isotropic.models["agglomerative_average"]
    assert isinstance(hierarchy, AgglomerativeClustering)
    artifacts["dendrogram"] = dendrogram_plot(
        hierarchy.linkage_matrix_,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/agglomerative_dendrogram.png",
    )
    artifacts["hierarchy_diagnostics"] = hierarchy_diagnostics_plot(
        hierarchy.linkage_matrix_,
        selected_clusters=hierarchy.n_clusters,
        context=context_label,
        artifacts=context.artifacts,
        relative_path="plots/agglomerative_diagnostics.png",
    )
    return artifacts


class ScratchClusteringBenchmark:
    """Compare clustering families using only internal and stability evidence."""

    def __init__(self, config: ScratchClusteringBenchmarkConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        datasets = clustering_suite(n_samples=self._config.n_samples, seed=context.seed)
        specs = _cluster_specs(self._config)
        runs = _fit_clustering_datasets(self._config, datasets, specs)
        stability = _stability_trials(self._config, runs, specs)
        candidates = _aggregate_cluster_candidates(runs, specs, stability)
        selected_name = select_highest(
            {candidate.name: candidate.metrics["selection_score"] for candidate in candidates}
        )
        selected = next(candidate for candidate in candidates if candidate.name == selected_name)

        records_path, csv_path, stability_path, convergence_path = _write_clustering_records(
            context, runs, specs, stability
        )
        assignments_path = _write_assignment_archive(context, runs, specs)
        diagnostics_path = _write_clustering_diagnostics(context, runs, specs)
        model_path = "models/selected_clustering_models.joblib"
        with context.artifacts.atomic_target(model_path) as temporary:
            joblib.dump(
                {
                    run.dataset.name: {
                        "scaler": run.scaler,
                        "model": run.models[selected_name],
                        "training_assignments": run.assignments[selected_name],
                    }
                    for run in runs
                },
                temporary,
            )
        plots = _clustering_plots(self._config, context, runs, specs, stability)
        metric_diagnostics_path, metric_plot_path = _metric_diagnostics(self._config, context)
        plots["random_partition_ari"] = metric_plot_path
        fingerprint_arrays = tuple(
            array for dataset in datasets for array in (dataset.features, dataset.evaluation_labels)
        )
        return RunResult(
            experiment=self._config.experiment,
            paradigm=LearningParadigm.UNSUPERVISED,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.GENERATOR,
                name="NumPy heterogeneous structure suite",
                version="sprint-03-v1",
                fingerprint_sha256=array_fingerprint(*fingerprint_arrays),
                target_used_for_fit=False,
                details={
                    "datasets": [dataset.name for dataset in datasets],
                    "samples_per_dataset": self._config.n_samples,
                    "labels_reserved_for_retrospective_evaluation": True,
                    "selection_inputs": ["silhouette", "assigned coverage", "stability"],
                },
            ),
            selected_model=selected_name,
            metrics=selected.metrics,
            candidates=candidates,
            artifacts={
                "candidate_records_json": records_path,
                "candidate_records_csv": csv_path,
                "stability_trials": stability_path,
                "convergence": convergence_path,
                "assignments": assignments_path,
                "learned_state_diagnostics": diagnostics_path,
                "selected_models": model_path,
                "metric_diagnostics": metric_diagnostics_path,
                **plots,
            },
            notes=(
                "Selection uses features, internal geometry, coverage, and named perturbations.",
                "ARI/NMI and truth-colored plots are retrospective and cannot influence selection.",
                "There is no universal winner across incompatible notions of cluster structure.",
                "DBSCAN and agglomerative assignments are transductive; k-means/GMM are inductive.",
            ),
        )


def _write_representation_records(
    context: RunContext,
    metrics: Mapping[str, Mapping[str, float]],
    seeds: Mapping[str, int],
    parameters: Mapping[str, Mapping[str, JsonValue]],
) -> tuple[str, str]:
    json_path = "metrics/representation_candidates.json"
    context.artifacts.write_json(
        json_path,
        [
            {
                "candidate": name,
                "seed": seeds[name],
                "parameters": parameters[name],
                "metrics": candidate_metrics,
            }
            for name, candidate_metrics in metrics.items()
        ],
    )
    csv_path = "metrics/representation_candidates.csv"
    metric_names = sorted(
        {metric for candidate_metrics in metrics.values() for metric in candidate_metrics}
    )
    with context.artifacts.atomic_target(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["candidate", "seed", "parameters_json", *metric_names],
            )
            writer.writeheader()
            for name, candidate_metrics in metrics.items():
                writer.writerow(
                    {
                        "candidate": name,
                        "seed": seeds[name],
                        "parameters_json": json.dumps(parameters[name], sort_keys=True),
                        **candidate_metrics,
                    }
                )
    return json_path, csv_path


class ScratchRepresentationBenchmark:
    """Compare linear and nonlinear representations without label-based tuning."""

    def __init__(self, config: ScratchRepresentationBenchmarkConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        data_seed = derive_named_seed(context.seed, self._config.experiment, "data")
        generated = make_blobs(
            n_samples=self._config.n_samples,
            n_features=self._config.n_features,
            centers=self._config.clusters,
            cluster_std=0.72,
            seed=data_seed,
        )
        scaler = StandardScaler()
        features = scaler.fit_transform(generated.features)
        pca_seed = derive_named_seed(context.seed, self._config.experiment, "candidate", "pca")
        tsne_seed = derive_named_seed(context.seed, self._config.experiment, "candidate", "tsne")
        pca = PCA(n_components=self._config.pca_components)
        pca_embedding = pca.fit_transform(features)
        tsne = TSNE(
            n_components=2,
            perplexity=self._config.tsne_perplexity,
            max_iter=self._config.tsne_iterations,
            early_exaggeration_iter=min(100, self._config.tsne_iterations // 3),
            init="pca",
            seed=tsne_seed,
        )
        try:
            tsne_embedding = tsne.fit_transform(features)
        except (MemoryError, RuntimeError, TypeError, ValueError) as error:
            raise UnsupervisedCandidateError("tsne", "representation_blobs", "fit") from error
        embeddings = {"pca": pca_embedding, "tsne": tsne_embedding}
        parameters: dict[str, Mapping[str, JsonValue]] = {
            "pca": {
                "n_components": self._config.pca_components,
                "whiten": False,
                "solver": "thin_svd",
                "deterministic_sign_convention": True,
            },
            "tsne": {
                "n_components": 2,
                "perplexity": self._config.tsne_perplexity,
                "learning_rate": 100.0,
                "max_iter": self._config.tsne_iterations,
                "early_exaggeration": 12.0,
                "early_exaggeration_iter": min(100, self._config.tsne_iterations // 3),
                "init": "pca",
            },
        }
        seeds = {"pca": pca_seed, "tsne": tsne_seed}
        metrics: dict[str, dict[str, float]] = {}
        for name, embedding in embeddings.items():
            recovery_seed = derive_named_seed(
                context.seed, self._config.experiment, "retrospective", name
            )
            recovered = KMeans(
                self._config.clusters,
                n_init=8,
                random_state=recovery_seed,
            ).fit_predict(embedding)
            neighbor_score = neighborhood_preservation(
                features,
                embedding,
                n_neighbors=min(10, len(features) - 1),
            )
            correlation = distance_correlation(features, embedding)
            metrics[name] = {
                "selection_score": float(0.8 * neighbor_score + 0.2 * max(correlation, 0.0)),
                "neighborhood_preservation": neighbor_score,
                "distance_correlation": correlation,
                "retrospective_ari": adjusted_rand_score(generated.targets, recovered),
                "retrospective_nmi": normalized_mutual_info_score(generated.targets, recovered),
            }
        metrics["pca"]["explained_variance_ratio"] = float(np.sum(pca.explained_variance_ratio_))
        metrics["tsne"]["final_kl_divergence"] = tsne.kl_divergence_
        metrics["tsne"]["perplexity_max_error"] = float(
            np.max(np.abs(tsne.perplexities_ - self._config.tsne_perplexity))
        )
        candidates = tuple(
            CandidateResult(name=name, metrics=value) for name, value in metrics.items()
        )
        selected_name = select_highest(
            {candidate.name: candidate.metrics["selection_score"] for candidate in candidates}
        )
        selected = next(candidate for candidate in candidates if candidate.name == selected_name)

        records_json, records_csv = _write_representation_records(
            context, metrics, seeds, parameters
        )
        embedding_path = "models/representations.npz"
        with context.artifacts.atomic_target(embedding_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    pca=pca_embedding,
                    tsne=tsne_embedding,
                    pca_mean=pca.mean_,
                    pca_components=pca.components_,
                    pca_singular_values=pca.singular_values_,
                    pca_explained_variance=pca.explained_variance_,
                    pca_explained_variance_ratio=pca.explained_variance_ratio_,
                    tsne_conditional_probabilities=tsne.conditional_probabilities_,
                    tsne_joint_probabilities=tsne.joint_probabilities_,
                    tsne_precisions=tsne.precisions_,
                    tsne_achieved_perplexities=tsne.perplexities_,
                    tsne_kl_history=np.asarray(tsne.kl_history_, dtype=np.float64),
                    tsne_gradient_norm_history=np.asarray(
                        tsne.gradient_norm_history_, dtype=np.float64
                    ),
                )
        model_path = "models/representation_models.joblib"
        with context.artifacts.atomic_target(model_path) as temporary:
            joblib.dump({"scaler": scaler, "pca": pca, "tsne": tsne}, temporary)
        tsne_diagnostics_path = "metrics/tsne_diagnostics.json"
        achieved_perplexities = tsne.perplexities_
        precisions = tsne.precisions_
        context.artifacts.write_json(
            tsne_diagnostics_path,
            {
                "target_perplexity": self._config.tsne_perplexity,
                "achieved_perplexity": {
                    "minimum": float(np.min(achieved_perplexities)),
                    "mean": float(np.mean(achieved_perplexities)),
                    "maximum": float(np.max(achieved_perplexities)),
                    "maximum_absolute_error": float(
                        np.max(np.abs(achieved_perplexities - self._config.tsne_perplexity))
                    ),
                },
                "affinity_precision": {
                    "minimum": float(np.min(precisions)),
                    "median": float(np.median(precisions)),
                    "maximum": float(np.max(precisions)),
                },
                "joint_probability_sum": float(np.sum(tsne.joint_probabilities_)),
                "joint_probability_symmetric": bool(
                    np.allclose(tsne.joint_probabilities_, tsne.joint_probabilities_.T)
                ),
                "initial_kl": tsne.initial_kl_,
                "post_exaggeration_start_kl": tsne.post_exaggeration_start_kl_,
                "final_kl": tsne.kl_divergence_,
                "kl_history": list(tsne.kl_history_),
                "gradient_norm_history": list(tsne.gradient_norm_history_),
                "iterations": tsne.n_iter_,
                "converged": tsne.converged_,
                "stop_reason": tsne.stop_reason_,
            },
        )

        full_pca = PCA().fit(features)
        full_coordinates = full_pca.transform(features)
        reconstruction_errors: list[float] = []
        for retained in range(1, full_pca.n_components_ + 1):
            reconstructed = (
                full_coordinates[:, :retained] @ full_pca.components_[:retained] + full_pca.mean_
            )
            reconstruction_errors.append(float(np.mean((features - reconstructed) ** 2)))
        context_label = (
            f"root seed={context.seed}; n={len(features)}; d={features.shape[1]}; "
            "targets excluded from fit and selection"
        )
        plots = {
            "embedding_comparison": embedding_comparison_plot(
                embeddings,
                generated.targets,
                context=context_label,
                artifacts=context.artifacts,
                relative_path="plots/representation_embeddings.png",
            ),
            "pca_diagnostics": pca_diagnostics_plot(
                full_pca.explained_variance_ratio_.tolist(),
                reconstruction_errors,
                context=context_label,
                artifacts=context.artifacts,
                relative_path="plots/pca_diagnostics.png",
            ),
            "tsne_optimization": optimization_histories_plot(
                {"t-SNE KL": tsne.kl_history_},
                title="t-SNE unexaggerated KL trajectory",
                ylabel="KL(P || Q)",
                context=context_label,
                artifacts=context.artifacts,
                relative_path="plots/tsne_optimization.png",
            ),
            "representation_metrics": representation_metrics_plot(
                {
                    name: {
                        "neighborhood_preservation": value["neighborhood_preservation"],
                        "distance_correlation": value["distance_correlation"],
                        "retrospective_ari": value["retrospective_ari"],
                    }
                    for name, value in metrics.items()
                },
                context=context_label,
                artifacts=context.artifacts,
                relative_path="plots/representation_metrics.png",
            ),
            "tsne_affinity_diagnostics": tsne_affinity_diagnostics_plot(
                tsne.perplexities_.tolist(),
                tsne.precisions_.tolist(),
                tsne.gradient_norm_history_,
                target_perplexity=self._config.tsne_perplexity,
                context=context_label,
                artifacts=context.artifacts,
                relative_path="plots/tsne_affinity_diagnostics.png",
            ),
        }

        perplexity_rows: list[list[float]] = [[], []]
        seed_rows: list[list[float]] = [[], []]
        sensitivity_perplexities = tuple(
            sorted(
                {max(3.0, self._config.tsne_perplexity * factor) for factor in (0.70, 1.0, 1.30)}
            )
        )
        sensitivity_payload: list[dict[str, JsonValue]] = []
        shared_perplexity_seed = derive_named_seed(
            context.seed,
            self._config.experiment,
            "sensitivity",
            "pca-init-perplexity-study",
        )
        for perplexity in sensitivity_perplexities:
            model = TSNE(
                perplexity=perplexity,
                max_iter=self._config.tsne_iterations,
                early_exaggeration_iter=min(100, self._config.tsne_iterations // 3),
                init="pca",
                seed=shared_perplexity_seed,
            )
            try:
                embedded = model.fit_transform(features)
            except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                raise UnsupervisedCandidateError(
                    "tsne",
                    "representation_blobs",
                    f"perplexity sensitivity {perplexity:g}",
                ) from error
            neighbor_score = neighborhood_preservation(features, embedded, n_neighbors=10)
            retrospective = KMeans(
                self._config.clusters,
                n_init=6,
                random_state=derive_named_seed(
                    context.seed,
                    self._config.experiment,
                    "retrospective",
                    "sensitivity",
                    f"perplexity-{perplexity:.8g}",
                ),
            ).fit_predict(embedded)
            ari = adjusted_rand_score(generated.targets, retrospective)
            perplexity_rows[0].append(neighbor_score)
            perplexity_rows[1].append(ari)
            sensitivity_payload.append(
                {
                    "study": "perplexity",
                    "initialization": "pca",
                    "perplexity": perplexity,
                    "seed": shared_perplexity_seed,
                    "neighborhood_preservation": neighbor_score,
                    "retrospective_ari": ari,
                    "final_kl": model.kl_divergence_,
                    "converged": model.converged_,
                }
            )

        sensitivity_seeds = tuple(
            derive_named_seed(
                context.seed,
                self._config.experiment,
                "sensitivity",
                "random-init",
                f"trial-{trial}",
            )
            for trial in range(3)
        )
        for trial, sensitivity_seed in enumerate(sensitivity_seeds):
            model = TSNE(
                perplexity=self._config.tsne_perplexity,
                max_iter=self._config.tsne_iterations,
                early_exaggeration_iter=min(100, self._config.tsne_iterations // 3),
                init="random",
                seed=sensitivity_seed,
            )
            try:
                embedded = model.fit_transform(features)
            except (MemoryError, RuntimeError, TypeError, ValueError) as error:
                raise UnsupervisedCandidateError(
                    "tsne",
                    "representation_blobs",
                    f"random-initialization sensitivity trial {trial}",
                ) from error
            neighbor_score = neighborhood_preservation(features, embedded, n_neighbors=10)
            retrospective = KMeans(
                self._config.clusters,
                n_init=6,
                random_state=derive_named_seed(sensitivity_seed, "retrospective"),
            ).fit_predict(embedded)
            ari = adjusted_rand_score(generated.targets, retrospective)
            seed_rows[0].append(neighbor_score)
            seed_rows[1].append(ari)
            sensitivity_payload.append(
                {
                    "study": "seed",
                    "initialization": "random",
                    "trial": trial,
                    "perplexity": self._config.tsne_perplexity,
                    "seed": sensitivity_seed,
                    "neighborhood_preservation": neighbor_score,
                    "retrospective_ari": ari,
                    "final_kl": model.kl_divergence_,
                    "converged": model.converged_,
                }
            )
        sensitivity_path = "metrics/tsne_sensitivity.json"
        context.artifacts.write_json(sensitivity_path, sensitivity_payload)
        plots["tsne_sensitivity"] = tsne_sensitivity_plot(
            np.asarray(perplexity_rows, dtype=np.float64),
            tuple(f"perplexity_{value:g}" for value in sensitivity_perplexities),
            np.asarray(seed_rows, dtype=np.float64),
            tuple(f"trial_{trial}" for trial in range(len(sensitivity_seeds))),
            metric_labels=("neighborhood_preservation", "retrospective_ari"),
            context=context_label,
            artifacts=context.artifacts,
            relative_path="plots/tsne_sensitivity.png",
        )

        return RunResult(
            experiment=self._config.experiment,
            paradigm=LearningParadigm.UNSUPERVISED,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.GENERATOR,
                name="NumPy high-dimensional Gaussian blobs",
                version="sprint-03-v1",
                fingerprint_sha256=array_fingerprint(generated.features, generated.targets),
                target_used_for_fit=False,
                details={
                    "sample_count": len(generated.features),
                    "feature_count": generated.features.shape[1],
                    "clusters": self._config.clusters,
                    "generator_seed": data_seed,
                    "labels_reserved_for_retrospective_evaluation": True,
                    "selection_inputs": ["neighbor overlap", "distance correlation"],
                },
            ),
            selected_model=selected_name,
            metrics=selected.metrics,
            candidates=candidates,
            artifacts={
                "candidate_records_json": records_json,
                "candidate_records_csv": records_csv,
                "embeddings": embedding_path,
                "models": model_path,
                "tsne_diagnostics": tsne_diagnostics_path,
                "tsne_sensitivity_records": sensitivity_path,
                **plots,
            },
            notes=(
                "PCA is inductive; t-SNE is transductive and exposes no transform method.",
                "Neighbor overlap and distance correlation select the representation without labels.",
                "K-means ARI/NMI and plot colors use truth only after representations are fixed.",
                "t-SNE is quadratic and global distances or apparent cluster sizes are not interpreted.",
            ),
        )
