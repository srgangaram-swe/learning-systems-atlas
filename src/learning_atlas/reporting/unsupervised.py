"""Artifact-safe diagnostic plots for unsupervised structure discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from numpy.typing import NDArray

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.validation import FloatArray
from learning_atlas.reporting.plots import publish_figure

IntArray = NDArray[np.int64]


def _stamp(figure: Figure, context: str, *, bottom: float = 0.12) -> None:
    if not context.strip():
        msg = "plot context must not be empty"
        raise ValueError(msg)
    figure.subplots_adjust(bottom=max(figure.subplotpars.bottom, bottom))
    figure.text(0.5, 0.012, context, ha="center", va="bottom", fontsize=8, color="dimgray")


def assignment_grid(
    features: Mapping[str, FloatArray],
    assignments: Mapping[tuple[str, str], IntArray],
    candidates: Sequence[str],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Compare candidate partitions across heterogeneous structure."""

    dataset_names = tuple(features)
    figure, axes = plt.subplots(
        len(dataset_names),
        len(candidates),
        figsize=(3.0 * len(candidates), 2.7 * len(dataset_names)),
        squeeze=False,
    )
    for row, dataset_name in enumerate(dataset_names):
        values = features[dataset_name]
        for column, candidate in enumerate(candidates):
            axis = axes[row, column]
            labels = assignments[(dataset_name, candidate)]
            axis.scatter(values[:, 0], values[:, 1], c=labels, cmap="tab10", s=11, alpha=0.82)
            if row == 0:
                axis.set_title(candidate.replace("_", " "))
            if column == 0:
                axis.set_ylabel(dataset_name.replace("_", " "))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(alpha=0.12)
    figure.suptitle("Feature-only assignments across incompatible cluster geometry", y=0.995)
    _stamp(figure, context, bottom=0.08)
    return publish_figure(figure, artifacts, relative_path)


def score_heatmap(
    matrix: FloatArray,
    rows: Sequence[str],
    columns: Sequence[str],
    *,
    title: str,
    colorbar_label: str,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
    value_range: tuple[float, float] | None = None,
) -> str:
    """Publish an annotated candidate-by-dataset score matrix."""

    if matrix.shape != (len(rows), len(columns)):
        msg = "score matrix shape must match row and column labels"
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(max(7.5, len(columns) * 1.4), max(4.8, len(rows))))
    if value_range is None:
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    else:
        image = axis.imshow(
            matrix,
            cmap="viridis",
            aspect="auto",
            vmin=value_range[0],
            vmax=value_range[1],
        )
    for row in range(len(rows)):
        for column in range(len(columns)):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", color="white")
    axis.set(
        title=title,
        xticks=np.arange(len(columns)),
        xticklabels=[label.replace("_", " ") for label in columns],
        yticks=np.arange(len(rows)),
        yticklabels=[label.replace("_", " ") for label in rows],
    )
    axis.tick_params(axis="x", labelrotation=25)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    _stamp(figure, context, bottom=0.20)
    return publish_figure(figure, artifacts, relative_path)


def stability_plot(
    values: Mapping[str, Sequence[float]],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Show every perturbation-stability trial for each candidate."""

    names = tuple(values)
    samples = [tuple(values[name]) for name in names]
    figure, axis = plt.subplots(figsize=(max(10.0, len(names) * 2.1), 5.4))
    positions = np.arange(1, len(names) + 1)
    axis.boxplot(samples, positions=positions, widths=0.55, patch_artist=True)
    for position, trials in zip(positions, samples, strict=True):
        axis.scatter(
            position + np.linspace(-0.08, 0.08, len(trials)),
            trials,
            s=26,
            alpha=0.8,
        )
    axis.set(
        title="Partition stability under named feature perturbations",
        ylabel="Adjusted Rand agreement with unperturbed assignment",
        ylim=(-0.05, 1.05),
        xticks=positions,
        xticklabels=[name.replace("_", " ") for name in names],
    )
    axis.tick_params(axis="x", labelrotation=24)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.grid(axis="y", alpha=0.2)
    _stamp(figure, context, bottom=0.27)
    return publish_figure(figure, artifacts, relative_path)


def optimization_histories_plot(
    histories: Mapping[str, Sequence[float]],
    *,
    title: str,
    ylabel: str,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Compare convergence histories without hiding failed or short runs."""

    if len(histories) == 1:
        figure, axis = plt.subplots(figsize=(8.4, 5.0))
        name, history = next(iter(histories.items()))
        axis.plot(
            np.arange(len(history)),
            history,
            marker="o",
            label=name.replace("_", " "),
            linewidth=1.8,
        )
        axis.set(title=title, xlabel="Iteration", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    else:
        column_count = 2
        row_count = int(np.ceil(len(histories) / column_count))
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(11.8, 4.3 * row_count),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for index, (axis, (name, history)) in enumerate(
            zip(flat_axes, histories.items(), strict=False)
        ):
            axis.plot(np.arange(len(history)), history, marker="o", linewidth=1.8)
            axis.set(
                title=name.replace("_", " "),
                ylabel=ylabel if index % column_count == 0 else None,
            )
            if index // column_count == row_count - 1:
                axis.set_xlabel("Iteration")
            else:
                axis.tick_params(axis="x", labelbottom=False)
            if len(history) == 1:
                axis.set_xlim(-0.5, 0.5)
                axis.set_xticks([0])
            axis.grid(alpha=0.2)
        for axis in flat_axes[len(histories) :]:
            axis.set_visible(False)
        figure.suptitle(title)
        figure.subplots_adjust(hspace=0.34, wspace=0.22)
    _stamp(figure, context, bottom=0.09)
    return publish_figure(figure, artifacts, relative_path)


def restart_objectives_plot(
    kmeans_inertias: Mapping[str, Sequence[float]],
    mixture_lower_bounds: Mapping[str, Sequence[float]],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Expose every restart objective instead of reporting only the winner."""

    dataset_names = tuple(kmeans_inertias)
    if dataset_names != tuple(mixture_lower_bounds):
        msg = "k-means and mixture restart mappings must have the same dataset order"
        raise ValueError(msg)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    positions = np.arange(len(dataset_names), dtype=np.float64)
    panels = (
        (axes[0], kmeans_inertias, "K-means++ restart outcomes", "Final inertia (lower is better)"),
        (
            axes[1],
            mixture_lower_bounds,
            "Gaussian-mixture restart outcomes",
            "Final mean log likelihood (higher is better)",
        ),
    )
    for axis, values, title, ylabel in panels:
        for position, dataset_name in zip(positions, dataset_names, strict=True):
            outcomes = np.asarray(values[dataset_name], dtype=np.float64)
            offsets = np.linspace(-0.11, 0.11, len(outcomes))
            axis.scatter(
                position + offsets,
                outcomes,
                s=38,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
            )
            axis.hlines(
                np.median(outcomes),
                position - 0.18,
                position + 0.18,
                color="black",
                linewidth=1.2,
            )
        axis.set(
            title=title,
            ylabel=ylabel,
            xticks=positions,
            xticklabels=[name.replace("_", " ") for name in dataset_names],
        )
        axis.tick_params(axis="x", labelrotation=24)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Named, order-independent restart streams; bars mark medians")
    _stamp(figure, context, bottom=0.21)
    return publish_figure(figure, artifacts, relative_path)


def mixture_density_plot(
    features: FloatArray,
    assignments: IntArray,
    means: FloatArray,
    covariances: FloatArray,
    weights: FloatArray,
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Plot the fitted two-dimensional mixture density and covariance ellipses."""

    if features.ndim != 2 or features.shape[1] != 2:
        msg = "mixture density plot requires a two-dimensional feature matrix"
        raise ValueError(msg)
    if means.shape != (len(weights), 2) or covariances.shape != (len(weights), 2, 2):
        msg = "mixture parameter shapes are inconsistent"
        raise ValueError(msg)

    margin = 0.15 * np.ptp(features, axis=0)
    lower = np.min(features, axis=0) - margin
    upper = np.max(features, axis=0) + margin
    x_grid = np.linspace(lower[0], upper[0], 180)
    y_grid = np.linspace(lower[1], upper[1], 180)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    density = np.zeros(len(points), dtype=np.float64)
    normalizer = 2.0 * np.pi
    for weight, mean, covariance in zip(weights, means, covariances, strict=True):
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0.0:
            msg = "mixture covariance must be positive definite"
            raise ValueError(msg)
        difference = points - mean
        solved = np.linalg.solve(covariance, difference.T).T
        mahalanobis = np.sum(difference * solved, axis=1)
        density += weight * np.exp(-0.5 * (mahalanobis + log_determinant)) / normalizer

    figure, axis = plt.subplots(figsize=(8.2, 6.0))
    contours = axis.contourf(
        grid_x,
        grid_y,
        density.reshape(grid_x.shape),
        levels=18,
        cmap="Blues",
        alpha=0.75,
    )
    axis.scatter(
        features[:, 0],
        features[:, 1],
        c=assignments,
        cmap="tab10",
        s=16,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.82,
    )
    colors = plt.get_cmap("tab10")
    for component, (mean, covariance) in enumerate(zip(means, covariances, strict=True)):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
        for standard_deviations, alpha in ((1.0, 0.9), (2.0, 0.65)):
            width, height = 2.0 * standard_deviations * np.sqrt(eigenvalues)
            axis.add_patch(
                Ellipse(
                    xy=mean,
                    width=float(width),
                    height=float(height),
                    angle=angle,
                    facecolor="none",
                    edgecolor=colors(component % 10),
                    linewidth=1.8,
                    alpha=alpha,
                )
            )
        axis.scatter(
            [mean[0]],
            [mean[1]],
            marker="X",
            s=100,
            color=colors(component % 10),
            edgecolor="black",
        )
    figure.colorbar(contours, ax=axis, label="Fitted mixture density")
    axis.set(
        title="Full-covariance Gaussian mixture: density and 1-SD/2-SD ellipses",
        xlabel="standardized feature 1",
        ylabel="standardized feature 2",
    )
    axis.grid(alpha=0.12)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def dbscan_sensitivity_plot(
    eps_values: Sequence[float],
    silhouette: Sequence[float],
    noise_fraction: Sequence[float],
    clusters: Sequence[int],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Expose DBSCAN hyperparameter sensitivity and coverage tradeoffs."""

    figure, left = plt.subplots(figsize=(8.4, 5.0))
    left.plot(eps_values, silhouette, marker="o", label="assigned silhouette")
    left.plot(eps_values, noise_fraction, marker="s", label="noise fraction")
    left.set(title="DBSCAN sensitivity on two moons", xlabel="eps", ylabel="Score / fraction")
    left.set_ylim(-0.05, 1.05)
    right = left.twinx()
    right.step(eps_values, clusters, where="mid", color="tab:red", label="clusters")
    right.set_ylabel("Clusters found")
    handles, labels = left.get_legend_handles_labels()
    extra_handles, extra_labels = right.get_legend_handles_labels()
    left.legend(handles + extra_handles, labels + extra_labels, loc="best")
    left.grid(alpha=0.2)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def k_distance_plot(
    k_distances: Sequence[float],
    *,
    eps: float,
    min_samples: int,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Plot sorted DBSCAN neighbor radii with the configured epsilon."""

    ordered = np.sort(np.asarray(k_distances, dtype=np.float64))
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.plot(np.arange(1, len(ordered) + 1), ordered, linewidth=2.0)
    axis.axhline(eps, color="tab:red", linestyle="--", label=f"configured eps={eps:g}")
    axis.set(
        title=f"DBSCAN {min_samples}-distance diagnostic on two moons",
        xlabel="Observation rank",
        ylabel=f"Distance to neighbor rank {min_samples}",
    )
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def dendrogram_plot(
    linkage: FloatArray,
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Render a compact dendrogram directly from a standard linkage matrix."""

    leaf_count = linkage.shape[0] + 1
    children = {
        leaf_count + offset: (int(row[0]), int(row[1])) for offset, row in enumerate(linkage)
    }

    def ordered_leaves(node: int) -> tuple[int, ...]:
        if node < leaf_count:
            return (node,)
        left, right = children[node]
        return (*ordered_leaves(left), *ordered_leaves(right))

    traversal = ordered_leaves(2 * leaf_count - 2)
    centers: dict[int, float] = {leaf: float(position) for position, leaf in enumerate(traversal)}
    heights: dict[int, float] = {index: 0.0 for index in range(leaf_count)}
    figure, axis = plt.subplots(figsize=(10.0, 5.0))
    for offset, row in enumerate(linkage):
        left_index, right_index = int(row[0]), int(row[1])
        height = float(row[2])
        left_x, right_x = centers[left_index], centers[right_index]
        axis.plot([left_x, left_x], [heights[left_index], height], color="tab:blue", linewidth=1.0)
        axis.plot(
            [right_x, right_x], [heights[right_index], height], color="tab:blue", linewidth=1.0
        )
        axis.plot([left_x, right_x], [height, height], color="tab:blue", linewidth=1.0)
        new_index = leaf_count + offset
        centers[new_index] = (left_x + right_x) / 2.0
        heights[new_index] = height
    axis.set(
        title="Average-linkage hierarchy with deterministic tree traversal",
        xlabel="Observations in dendrogram leaf order",
        ylabel="Merge distance",
    )
    axis.set_xticks([])
    axis.grid(axis="y", alpha=0.2)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def hierarchy_diagnostics_plot(
    linkage: FloatArray,
    *,
    selected_clusters: int,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Expose cut sensitivity and deterministic cubic-work growth."""

    sample_count = len(linkage) + 1
    merge_steps = np.arange(1, sample_count)
    selected_step = sample_count - selected_clusters
    sizes = np.unique(
        np.geomspace(8, max(8, sample_count), num=min(14, max(2, sample_count - 7))).astype(int)
    )
    pair_evaluations = np.asarray(
        [sum(active * (active - 1) // 2 for active in range(2, int(size) + 1)) for size in sizes],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    axes[0].plot(merge_steps, linkage[:, 2], linewidth=1.9)
    axes[0].axvline(
        selected_step,
        color="tab:red",
        linestyle="--",
        label=f"configured cut: k={selected_clusters}",
    )
    axes[0].set(
        title="Cut sensitivity along the merge hierarchy",
        xlabel="Chronological merge",
        ylabel="Merge distance",
    )
    axes[0].legend(loc="best")
    axes[1].loglog(sizes, pair_evaluations, marker="o", linewidth=1.9)
    axes[1].set(
        title="Clarity-first search work (deterministic proxy)",
        xlabel="Samples",
        ylabel="Candidate cluster-pair evaluations",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Average linkage: threshold sensitivity and O(n^3) work envelope")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def random_partition_ari_plot(
    scores: Sequence[float],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Show the adjusted Rand baseline over named independent partitions."""

    values = np.asarray(scores, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.hist(values, bins=16, alpha=0.78, color="tab:purple", edgecolor="white")
    axis.axvline(0.0, color="black", linestyle="--", label="chance expectation = 0")
    axis.axvline(
        float(np.mean(values)),
        color="tab:red",
        label=f"observed mean = {np.mean(values):.4f}",
    )
    axis.set(
        title="Adjusted Rand index under independent random partitions",
        xlabel="ARI",
        ylabel="Trial count",
    )
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.2)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def pca_diagnostics_plot(
    explained_ratio: Sequence[float],
    reconstruction_error: Sequence[float],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Show retained variance and reconstruction error across component counts."""

    component_numbers = np.arange(1, len(explained_ratio) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].bar(component_numbers, explained_ratio, alpha=0.75, label="individual")
    axes[0].plot(component_numbers, np.cumsum(explained_ratio), marker="o", label="cumulative")
    axes[0].set(title="PCA explained variance", xlabel="Components", ylabel="Variance ratio")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[1].plot(component_numbers, reconstruction_error, marker="o", color="tab:orange")
    axes[1].set(
        title="Reconstruction improves with retained rank",
        xlabel="Components",
        ylabel="Mean squared reconstruction error",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def embedding_comparison_plot(
    embeddings: Mapping[str, FloatArray],
    evaluation_labels: IntArray,
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Display representations with truth used only as retrospective coloring."""

    figure, axes = plt.subplots(1, len(embeddings), figsize=(5.0 * len(embeddings), 4.6))
    for axis, (name, embedding) in zip(np.atleast_1d(axes), embeddings.items(), strict=True):
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=evaluation_labels,
            cmap="tab10",
            s=20,
            alpha=0.82,
        )
        axis.set(title=name.replace("_", " "), xlabel="embedding 1", ylabel="embedding 2")
        axis.grid(alpha=0.15)
    figure.suptitle("Retrospective coloring only; targets never enter representation fitting")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def representation_metrics_plot(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Compare label-free neighborhood/distance preservation and external recovery."""

    names = tuple(metrics)
    metric_names = tuple(next(iter(metrics.values())))
    positions = np.arange(len(names))
    width = 0.8 / len(metric_names)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    for index, metric_name in enumerate(metric_names):
        offset = (index - (len(metric_names) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            [metrics[name][metric_name] for name in names],
            width,
            label=metric_name.replace("_", " "),
        )
    axis.set(
        title="Representation evidence: internal selection and retrospective recovery",
        ylabel="Score",
        ylim=(-0.05, 1.05),
        xticks=positions,
        xticklabels=[name.replace("_", " ") for name in names],
    )
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.2)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def tsne_affinity_diagnostics_plot(
    achieved_perplexities: Sequence[float],
    precisions: Sequence[float],
    gradient_norms: Sequence[float],
    *,
    target_perplexity: float,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Expose per-row affinity search and optimization-scale diagnostics."""

    achieved = np.asarray(achieved_perplexities, dtype=np.float64)
    precision = np.asarray(precisions, dtype=np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7))
    axes[0].plot(achieved - target_perplexity, linewidth=1.2)
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].set(
        title="Per-row perplexity search error",
        xlabel="Observation row",
        ylabel="Achieved - target perplexity",
    )
    axes[1].hist(precision, bins=18, color="tab:green", edgecolor="white", alpha=0.78)
    axes[1].set(
        title="Gaussian affinity precision",
        xlabel="Binary-search precision",
        ylabel="Row count",
    )
    axes[2].semilogy(np.arange(len(gradient_norms)), gradient_norms, linewidth=1.6)
    axes[2].set(
        title="Gradient norm through optimization",
        xlabel="Iteration",
        ylabel="L2 norm (log scale)",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("t-SNE affinity calibration and convergence diagnostics")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def tsne_sensitivity_plot(
    perplexity_scores: FloatArray,
    perplexity_labels: Sequence[str],
    seed_scores: FloatArray,
    seed_labels: Sequence[str],
    *,
    metric_labels: Sequence[str],
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Separate t-SNE perplexity effects from random-initialization effects."""

    expected_perplexity = (len(metric_labels), len(perplexity_labels))
    expected_seed = (len(metric_labels), len(seed_labels))
    if perplexity_scores.shape != expected_perplexity or seed_scores.shape != expected_seed:
        msg = "t-SNE sensitivity matrices must match their metric and trial labels"
        raise ValueError(msg)
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    panels = (
        (axes[0], perplexity_scores, perplexity_labels, "PCA-init perplexity sensitivity"),
        (axes[1], seed_scores, seed_labels, "Random-init seed sensitivity"),
    )
    image = None
    for panel_index, (axis, matrix, column_labels, title) in enumerate(panels):
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        for row in range(len(metric_labels)):
            for column in range(len(column_labels)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        axis.set(
            title=title,
            xticks=np.arange(len(column_labels)),
            xticklabels=column_labels,
            yticks=np.arange(len(metric_labels)),
        )
        if panel_index == 0:
            axis.set_yticklabels([label.replace("_", " ") for label in metric_labels])
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.tick_params(axis="x", labelrotation=24)
    assert image is not None
    figure.subplots_adjust(left=0.18, right=0.89, wspace=0.20)
    colorbar_axis = figure.add_axes((0.92, 0.20, 0.018, 0.62))
    figure.colorbar(image, cax=colorbar_axis, label="Score")
    figure.suptitle("t-SNE sensitivity; ARI remains retrospective")
    _stamp(figure, context, bottom=0.18)
    return publish_figure(figure, artifacts, relative_path)
