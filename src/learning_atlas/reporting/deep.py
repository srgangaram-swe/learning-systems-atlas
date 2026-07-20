"""Artifact-safe diagnostic plots for the deep-learning studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.validation import FloatArray
from learning_atlas.reporting.plots import publish_figure

IntArray = NDArray[np.int64]


def _validate_context(context: str) -> None:
    if not context.strip():
        msg = "plot context must not be empty"
        raise ValueError(msg)


def _finite_series(values: Sequence[float], *, context: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        msg = f"{context} must contain at least one history point"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{context} must contain only finite values"
        raise ValueError(msg)
    return array


def _stamp(figure: Figure, context: str, *, bottom: float = 0.18) -> None:
    _validate_context(context)
    figure.subplots_adjust(bottom=max(figure.subplotpars.bottom, bottom))
    figure.text(0.5, 0.012, context, ha="center", va="bottom", fontsize=8, color="dimgray")


@dataclass(frozen=True, slots=True)
class DecisionPanel:
    """One planar task with a probability field evaluated on a dense grid."""

    features: FloatArray
    targets: IntArray
    grid_x: FloatArray
    grid_y: FloatArray
    positive_probability: FloatArray


def scratch_loss_curves_plot(
    histories: Mapping[str, Sequence[float]],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish per-task training-loss trajectories of the autograd MLP."""

    if not histories:
        msg = "loss-curve plot requires at least one history"
        raise ValueError(msg)
    _validate_context(context)
    validated = {
        name: _finite_series(history, context=f"history {name!r}")
        for name, history in histories.items()
    }
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for name, history in validated.items():
        axis.plot(range(1, len(history) + 1), history, label=name.replace("_", " "), lw=1.6)
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean training cross-entropy")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Autograd MLP optimization on linearly inseparable tasks")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def decision_boundary_grid(
    panels: Mapping[str, DecisionPanel],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish learned probability fields with the training points overlaid."""

    if not panels:
        msg = "decision-boundary plot requires at least one panel"
        raise ValueError(msg)
    _validate_context(context)
    for name, panel in panels.items():
        if panel.features.ndim != 2 or panel.features.shape[1] != 2:
            msg = f"decision panel {name!r} requires two feature columns"
            raise ValueError(msg)
        if panel.features.shape[0] == 0 or panel.targets.shape != (len(panel.features),):
            msg = f"decision panel {name!r} requires one target per feature row"
            raise ValueError(msg)
        if (
            panel.grid_x.ndim != 2
            or panel.grid_x.shape != panel.grid_y.shape
            or panel.grid_x.shape != panel.positive_probability.shape
            or panel.grid_x.size == 0
        ):
            msg = f"decision panel {name!r} grid and probability shapes must align"
            raise ValueError(msg)
        arrays = (
            panel.features,
            panel.targets,
            panel.grid_x,
            panel.grid_y,
            panel.positive_probability,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            msg = f"decision panel {name!r} requires finite values"
            raise ValueError(msg)
        if np.any((panel.positive_probability < 0.0) | (panel.positive_probability > 1.0)):
            msg = f"decision panel {name!r} probabilities must lie in [0, 1]"
            raise ValueError(msg)
    figure, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4), squeeze=False)
    for axis, (name, panel) in zip(axes[0], panels.items(), strict=True):
        field = axis.contourf(
            panel.grid_x,
            panel.grid_y,
            panel.positive_probability,
            levels=21,
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            alpha=0.85,
        )
        axis.contour(
            panel.grid_x,
            panel.grid_y,
            panel.positive_probability,
            levels=[0.5],
            colors="black",
            linewidths=1.2,
        )
        axis.scatter(
            panel.features[:, 0],
            panel.features[:, 1],
            c=panel.targets,
            cmap="coolwarm",
            s=14,
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        figure.colorbar(field, ax=axis, label="P(class 1)")
    figure.suptitle("From-scratch MLP decision boundaries", y=0.98)
    _stamp(figure, context, bottom=0.16)
    return publish_figure(figure, artifacts, relative_path)


def training_curves_plot(
    curves: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    ylabel: str,
    title: str,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
    log_scale: bool = False,
) -> str:
    """Publish per-candidate train/validation trajectories from Trainer histories."""

    if not curves:
        msg = "training-curve plot requires at least one candidate"
        raise ValueError(msg)
    _validate_context(context)
    validated_curves: dict[str, dict[str, FloatArray]] = {}
    for candidate, series_by_name in curves.items():
        if not series_by_name:
            msg = f"candidate {candidate!r} has no series to plot"
            raise ValueError(msg)
        validated_curves[candidate] = {
            name: _finite_series(series, context=f"series {candidate!r}/{name!r}")
            for name, series in series_by_name.items()
        }
        if log_scale and any(
            np.any(series <= 0.0) for series in validated_curves[candidate].values()
        ):
            msg = "log-scale training curves require strictly positive values"
            raise ValueError(msg)
    figure, axes = plt.subplots(1, len(curves), figsize=(5.6 * len(curves), 4.2), squeeze=False)
    for axis, (candidate, validated_series_by_name) in zip(
        axes[0], validated_curves.items(), strict=True
    ):
        for series_name, series in validated_series_by_name.items():
            axis.plot(
                range(1, len(series) + 1),
                series,
                label=series_name.replace("_", " "),
                lw=1.6,
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(candidate.replace("_", " "))
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title, y=0.98)
    _stamp(figure, context, bottom=0.16)
    return publish_figure(figure, artifacts, relative_path)


def confusion_matrix_plot(
    matrix: IntArray,
    labels: Sequence[str],
    *,
    title: str,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish an annotated confusion matrix."""

    if matrix.shape != (len(labels), len(labels)):
        msg = "confusion matrix shape must match the label count"
        raise ValueError(msg)
    _validate_context(context)
    if len(labels) == 0:
        msg = "confusion matrix requires at least one label"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        msg = "confusion matrix counts must be finite and non-negative"
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("predicted label")
    axis.set_ylabel("true label")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    figure.colorbar(image, ax=axis, label="count")
    axis.set_title(title)
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def image_panel_plot(
    panels: Mapping[str, FloatArray],
    captions: Mapping[str, Sequence[str]],
    *,
    title: str,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish rows of square grayscale images with per-image captions."""

    if not panels:
        msg = "image panel requires at least one row"
        raise ValueError(msg)
    _validate_context(context)
    for name, images in panels.items():
        if (
            images.ndim != 3
            or images.shape[0] == 0
            or images.shape[1] == 0
            or images.shape[1] != images.shape[2]
        ):
            msg = f"image row {name!r} requires at least one image with square shape (n, h, h)"
            raise ValueError(msg)
        if not np.all(np.isfinite(images)):
            msg = f"image row {name!r} requires finite pixels"
            raise ValueError(msg)
        if len(captions.get(name, ())) != images.shape[0]:
            msg = f"captions for row {name!r} must match its image count"
            raise ValueError(msg)
    counts = {name: images.shape[0] for name, images in panels.items()}
    columns = max(counts.values())
    if columns == 0:
        msg = "image panel requires at least one image per row"
        raise ValueError(msg)
    figure, axes = plt.subplots(
        len(panels), columns, figsize=(1.35 * columns, 1.65 * len(panels)), squeeze=False
    )
    for row, (name, images) in enumerate(panels.items()):
        row_captions = captions.get(name, ())
        for column in range(columns):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column >= images.shape[0]:
                axis.axis("off")
                continue
            axis.imshow(images[column], cmap="gray_r")
            axis.set_title(row_captions[column], fontsize=7)
            if column == 0:
                axis.set_ylabel(name.replace("_", " "), fontsize=8)
    figure.suptitle(title, y=0.995)
    _stamp(figure, context, bottom=0.06)
    return publish_figure(figure, artifacts, relative_path)


def conv_filters_plot(
    filters: FloatArray,
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish learned first-layer convolution kernels."""

    if filters.ndim != 4 or filters.shape[1] != 1:
        msg = "expected single-channel filters shaped (count, 1, height, width)"
        raise ValueError(msg)
    count = filters.shape[0]
    _validate_context(context)
    if count == 0 or filters.shape[2] == 0 or filters.shape[3] == 0:
        msg = "filter plot requires at least one non-empty kernel"
        raise ValueError(msg)
    if not np.all(np.isfinite(filters)):
        msg = "filter plot requires finite kernel values"
        raise ValueError(msg)
    figure, axes = plt.subplots(1, count, figsize=(1.4 * count, 1.9), squeeze=False)
    for index in range(count):
        axis = axes[0, index]
        axis.imshow(filters[index, 0], cmap="RdBu_r")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"k{index}", fontsize=8)
    figure.suptitle("Learned first-layer convolution kernels", y=1.02)
    _stamp(figure, context, bottom=0.14)
    return publish_figure(figure, artifacts, relative_path)


def accuracy_by_length_plot(
    bucket_labels: Sequence[str],
    accuracies: Mapping[str, Sequence[float]],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish per-length-bucket accuracy for each sequence candidate."""

    if not accuracies:
        msg = "accuracy-by-length plot requires at least one candidate"
        raise ValueError(msg)
    _validate_context(context)
    if not bucket_labels:
        msg = "accuracy-by-length plot requires at least one non-empty bucket"
        raise ValueError(msg)
    validated_accuracies: dict[str, FloatArray] = {}
    for candidate, values in accuracies.items():
        if len(values) != len(bucket_labels):
            msg = f"candidate {candidate!r} must provide one accuracy per bucket"
            raise ValueError(msg)
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
            msg = f"candidate {candidate!r} accuracy values must lie in [0, 1]"
            raise ValueError(msg)
        validated_accuracies[candidate] = array
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    positions = np.arange(len(bucket_labels), dtype=np.float64)
    width = 0.8 / len(accuracies)
    for offset, (candidate, validated_values) in enumerate(validated_accuracies.items()):
        axis.bar(
            positions + offset * width,
            validated_values,
            width=width,
            label=candidate.replace("_", " "),
        )
    axis.axhline(0.5, color="dimgray", ls="--", lw=1.0, label="chance")
    axis.set_xticks(positions + 0.4 - width / 2.0, bucket_labels)
    axis.set_xlabel("true sequence length bucket")
    axis.set_ylabel("held-out accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.grid(alpha=0.25, axis="y")
    axis.set_title("Long-range accuracy by sequence length")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def error_histogram_plot(
    inlier_errors: FloatArray,
    anomaly_errors: FloatArray,
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish overlapping reconstruction-error distributions."""

    if inlier_errors.size == 0 or anomaly_errors.size == 0:
        msg = "error histogram requires non-empty inlier and anomaly errors"
        raise ValueError(msg)
    _validate_context(context)
    if inlier_errors.ndim != 1 or anomaly_errors.ndim != 1:
        msg = "error samples must be 1D one-dimensional arrays"
        raise ValueError(msg)
    if not np.all(np.isfinite(inlier_errors)) or not np.all(np.isfinite(anomaly_errors)):
        msg = "error samples must contain only finite values"
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    bins = np.histogram_bin_edges(np.concatenate([inlier_errors, anomaly_errors]), bins=40).tolist()
    axis.hist(inlier_errors, bins=bins, alpha=0.65, label="inliers", color="#2E7DAF")
    axis.hist(anomaly_errors, bins=bins, alpha=0.65, label="injected anomalies", color="#C44E52")
    axis.set_xlabel("per-sample mean-squared reconstruction error")
    axis.set_ylabel("count")
    axis.legend()
    axis.grid(alpha=0.25)
    axis.set_title("Reconstruction error separates structure from corruption")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def roc_curves_plot(
    curves: Mapping[str, tuple[FloatArray, FloatArray, float]],
    *,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish anomaly-detection ROC curves with their exact AUROC values."""

    if not curves:
        msg = "ROC plot requires at least one curve"
        raise ValueError(msg)
    _validate_context(context)
    for name, (false_positive_rates, true_positive_rates, auroc) in curves.items():
        if (
            false_positive_rates.ndim != 1
            or true_positive_rates.ndim != 1
            or false_positive_rates.shape != true_positive_rates.shape
            or false_positive_rates.size < 2
        ):
            msg = f"ROC curve {name!r} arrays must have the same one-dimensional shape"
            raise ValueError(msg)
        if (
            not np.all(np.isfinite(false_positive_rates))
            or not np.all(np.isfinite(true_positive_rates))
            or not np.isfinite(auroc)
        ):
            msg = f"ROC curve {name!r} must contain finite values"
            raise ValueError(msg)
        if (
            np.any((false_positive_rates < 0.0) | (false_positive_rates > 1.0))
            or np.any((true_positive_rates < 0.0) | (true_positive_rates > 1.0))
            or not 0.0 <= auroc <= 1.0
        ):
            msg = f"ROC curve {name!r} rates and AUROC must lie in [0, 1]"
            raise ValueError(msg)
        if np.any(np.diff(false_positive_rates) < 0.0) or np.any(
            np.diff(true_positive_rates) < 0.0
        ):
            msg = f"ROC curve {name!r} rates must be monotonic"
            raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(5.6, 5.2))
    for name, (false_positive_rates, true_positive_rates, auroc) in curves.items():
        axis.plot(
            false_positive_rates,
            true_positive_rates,
            lw=1.8,
            label=f"{name.replace('_', ' ')} (AUROC {auroc:.3f})",
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="dimgray", ls="--", lw=1.0, label="chance")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    axis.set_title("Anomaly scoring on held-out digits")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)


def latent_scatter_plot(
    latent: FloatArray,
    labels: IntArray,
    *,
    axis_label: str,
    context: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish the 2-D latent view colored by retrospective class labels."""

    if latent.ndim != 2 or latent.shape[1] != 2:
        msg = f"latent scatter requires (n, 2) coordinates; received {latent.shape}"
        raise ValueError(msg)
    _validate_context(context)
    if latent.shape[0] == 0:
        msg = "latent scatter requires at least one point"
        raise ValueError(msg)
    if labels.ndim != 1 or labels.shape[0] != latent.shape[0]:
        msg = "latent scatter requires one label per point"
        raise ValueError(msg)
    if not np.all(np.isfinite(latent)):
        msg = "latent scatter coordinates must be finite"
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    scatter = axis.scatter(latent[:, 0], latent[:, 1], c=labels, cmap="tab10", s=14, alpha=0.85)
    figure.colorbar(scatter, ax=axis, label="digit class (retrospective color only)")
    axis.set_xlabel(f"{axis_label} 1")
    axis.set_ylabel(f"{axis_label} 2")
    axis.grid(alpha=0.25)
    axis.set_title("Bottleneck latent space organizes classes without labels")
    _stamp(figure, context)
    return publish_figure(figure, artifacts, relative_path)
