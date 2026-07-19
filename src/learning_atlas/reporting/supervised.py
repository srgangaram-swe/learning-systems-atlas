"""Shared, artifact-safe visual evidence for from-scratch supervised models."""

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.validation import FloatArray
from learning_atlas.reporting.plots import publish_figure


class Predictor(Protocol):
    """Minimal plot-time prediction interface."""

    def predict(self, features: ArrayLike) -> FloatArray:
        """Return one prediction per row."""


def _stamp_context(figure: Figure, context_label: str, *, bottom: float = 0.15) -> None:
    """Attach compact experimental provenance to a published figure."""

    if not context_label.strip():
        msg = "plot context label must not be empty"
        raise ValueError(msg)
    figure.subplots_adjust(bottom=max(figure.subplotpars.bottom, bottom))
    figure.text(
        0.5,
        0.012,
        context_label,
        ha="center",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )


def cv_distribution_plot(
    names: Sequence[str],
    fold_scores: Sequence[Sequence[float]],
    *,
    metric_label: str,
    title: str,
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Show every validation fold rather than hiding variability in a mean."""

    figure, axis = plt.subplots(figsize=(max(9.0, len(names) * 0.85), 5.3))
    positions = np.arange(1, len(names) + 1)
    axis.boxplot(fold_scores, positions=positions, widths=0.55, patch_artist=True)
    for position, scores in zip(positions, fold_scores, strict=True):
        jitter = np.linspace(-0.09, 0.09, len(scores))
        axis.scatter(position + jitter, scores, s=20, alpha=0.78, zorder=3)
    axis.set(
        title=title,
        ylabel=metric_label,
        xticks=positions,
        xticklabels=[name.replace("_", " ") for name in names],
    )
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(axis="y", alpha=0.22)
    _stamp_context(figure, context_label, bottom=0.30)
    return publish_figure(figure, artifacts, relative_path)


def heldout_comparison_plot(
    names: Sequence[str],
    cv_means: Sequence[float],
    cv_stds: Sequence[float],
    test_values: Sequence[float],
    *,
    metric_label: str,
    title: str,
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Compare training-only selection evidence with the untouched holdout."""

    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(9.0, len(names) * 0.92), 5.3))
    axis.bar(
        positions - width / 2,
        cv_means,
        width,
        yerr=cv_stds,
        capsize=3,
        label="training-fold CV mean ± SD",
    )
    axis.bar(positions + width / 2, test_values, width, label="untouched holdout")
    axis.set(
        title=title,
        ylabel=metric_label,
        xticks=positions,
        xticklabels=[name.replace("_", " ") for name in names],
    )
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(axis="y", alpha=0.22)
    if len(names) >= 10:
        axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    else:
        axis.legend(fontsize=8)
    _stamp_context(figure, context_label, bottom=0.30)
    return publish_figure(figure, artifacts, relative_path)


def regression_diagnostics_plot(
    observed: FloatArray,
    predicted: FloatArray,
    *,
    model_name: str,
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish prediction, residual, and error-distribution diagnostics."""

    residuals = observed - predicted
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    lower = float(min(np.min(observed), np.min(predicted)))
    upper = float(max(np.max(observed), np.max(predicted)))
    axes[0].scatter(observed, predicted, alpha=0.7, edgecolor="none")
    axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axes[0].set(title="Observed vs. predicted", xlabel="Observed target", ylabel="Prediction")
    axes[1].scatter(predicted, residuals, alpha=0.7, edgecolor="none")
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set(title="Residual structure", xlabel="Prediction", ylabel="Observed - predicted")
    axes[2].hist(residuals, bins=18, alpha=0.82, edgecolor="white")
    axes[2].axvline(0.0, color="black", linestyle="--")
    axes[2].set(title="Residual distribution", xlabel="Residual", ylabel="Count")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"Untouched-test regression diagnostics — {model_name.replace('_', ' ')}")
    _stamp_context(figure, context_label, bottom=0.17)
    return publish_figure(figure, artifacts, relative_path)


def optimization_plot(
    curves: Mapping[str, Sequence[float]],
    *,
    title: str,
    ylabel: str,
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Plot finite iterative optimization or stagewise loss histories."""

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    for name, values in curves.items():
        if values:
            axis.plot(np.arange(1, len(values) + 1), values, label=name.replace("_", " "))
    axis.set(title=title, xlabel="Iteration / boosting stage", ylabel=ylabel)
    axis.set_yscale("log")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8)
    _stamp_context(figure, context_label, bottom=0.18)
    return publish_figure(figure, artifacts, relative_path)


def regularization_path_plot(
    alphas: Sequence[float],
    ridge_norms: Sequence[float],
    lasso_nonzero: Sequence[int],
    *,
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Expose shrinkage and exact sparsity over a prespecified alpha grid."""

    figure, left = plt.subplots(figsize=(8.5, 5.0))
    right = left.twinx()
    left.plot(alphas, ridge_norms, marker="o", label="Ridge coefficient norm")
    right.plot(alphas, lasso_nonzero, marker="s", color="tab:orange", label="Lasso nonzeros")
    left.set_xscale("log")
    left.set(
        title="Regularization path on the training partition",
        xlabel="Regularization strength alpha (log scale)",
        ylabel="Ridge ‖coefficient‖₂",
    )
    right.set_ylabel("Lasso nonzero coefficients")
    left.grid(alpha=0.22)
    lines = [*left.get_lines(), *right.get_lines()]
    left.legend(lines, ["Ridge coefficient norm", "Lasso nonzeros"], fontsize=8)
    _stamp_context(figure, context_label, bottom=0.18)
    return publish_figure(figure, artifacts, relative_path)


def _binary_roc(observed: FloatArray, probability: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(-probability, kind="stable")
    ordered = observed[order]
    ordered_probability = probability[order]
    positives = float(np.sum(ordered == 1.0))
    negatives = float(np.sum(ordered == 0.0))
    true_positive = np.cumsum(ordered == 1.0, dtype=np.float64)
    false_positive = np.cumsum(ordered == 0.0, dtype=np.float64)
    # Advance the curve only after a complete tied-score block. Emitting
    # intermediate points makes AUC depend on row order and can turn a constant
    # score into a fictitious perfect classifier.
    block_ends = np.flatnonzero(
        np.concatenate((ordered_probability[:-1] != ordered_probability[1:], [True]))
    )
    tpr = np.concatenate(([0.0], true_positive[block_ends] / positives))
    fpr = np.concatenate(([0.0], false_positive[block_ends] / negatives))
    return np.asarray(fpr), np.asarray(tpr)


def classification_diagnostics_plot(
    observed: FloatArray,
    predicted: FloatArray,
    probability_by_model: Mapping[str, FloatArray],
    *,
    selected_model: str,
    reliability_models: Sequence[str],
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Publish ROC, confusion, and reliability evidence for probabilistic models."""

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    for name, probability in probability_by_model.items():
        false_positive, true_positive = _binary_roc(observed, probability)
        axes[0].plot(false_positive, true_positive, label=name.replace("_", " "))
    axes[0].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--")
    axes[0].set(
        title="Untouched-test ROC", xlabel="False-positive rate", ylabel="True-positive rate"
    )
    axes[0].legend(fontsize=7)

    classes = np.asarray([0.0, 1.0])
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row, class_value in enumerate(classes):
        for column, predicted_value in enumerate(classes):
            matrix[row, column] = int(
                np.sum((observed == class_value) & (predicted == predicted_value))
            )
    axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1].set(
        title=f"Confusion — {selected_model.replace('_', ' ')}",
        xlabel="Predicted class",
        ylabel="Observed class",
        xticks=(0, 1),
        yticks=(0, 1),
    )

    plotted_reliability: set[str] = set()
    for name in reliability_models:
        if name in plotted_reliability:
            continue
        selected_probability = probability_by_model.get(name)
        if selected_probability is None:
            continue
        edges = np.linspace(0.0, 1.0, 9)
        predicted_bins: list[float] = []
        observed_bins: list[float] = []
        for lower, upper in pairwise(edges):
            include_upper = upper == 1.0
            mask = (selected_probability >= lower) & (
                selected_probability <= upper if include_upper else selected_probability < upper
            )
            if np.any(mask):
                predicted_bins.append(float(np.mean(selected_probability[mask])))
                observed_bins.append(float(np.mean(observed[mask])))
        axes[2].plot(
            predicted_bins,
            observed_bins,
            marker="o",
            linewidth=2.2 if name == selected_model else 1.5,
            label=name.replace("_", " "),
        )
        plotted_reliability.add(name)
    if plotted_reliability:
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "No requested model exposes probabilities", ha="center")
    axes[2].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--")
    axes[2].set(
        title="Reliability (descriptive, not calibration claim)",
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("From-scratch classification diagnostics")
    _stamp_context(figure, context_label, bottom=0.17)
    return publish_figure(figure, artifacts, relative_path)


def decision_boundary_plot(
    predictors: Mapping[str, Predictor],
    features: FloatArray,
    targets: FloatArray,
    reference_features: FloatArray,
    *,
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Project model decisions across two raw features while fixing all others at medians."""

    if reference_features.shape[1] != features.shape[1]:
        msg = "reference and displayed features must contain the same columns"
        raise ValueError(msg)
    selected = tuple(predictors.items())[:6]
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.8), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    x_margin = 0.15 * float(np.ptp(features[:, 0])) + 0.1
    y_margin = 0.15 * float(np.ptp(features[:, 1])) + 0.1
    x_values = np.linspace(
        float(np.min(features[:, 0])) - x_margin, float(np.max(features[:, 0])) + x_margin, 90
    )
    y_values = np.linspace(
        float(np.min(features[:, 1])) - y_margin, float(np.max(features[:, 1])) + y_margin, 90
    )
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    base = np.tile(np.median(reference_features, axis=0), (grid_x.size, 1))
    base[:, 0] = grid_x.ravel()
    base[:, 1] = grid_y.ravel()
    for axis, (name, predictor) in zip(flat_axes, selected, strict=False):
        decision = predictor.predict(base).reshape(grid_x.shape)
        axis.contourf(grid_x, grid_y, decision, levels=(-0.5, 0.5, 1.5), alpha=0.22)
        axis.scatter(features[:, 0], features[:, 1], c=targets, s=13, alpha=0.65, edgecolor="none")
        axis.set_title(name.replace("_", " "))
        axis.grid(alpha=0.15)
    for axis in flat_axes[len(selected) :]:
        axis.set_visible(False)
    figure.supxlabel(
        "Raw feature 0; remaining features fixed at training medians",
        y=0.05,
    )
    figure.supylabel("Raw feature 1")
    figure.suptitle("Untouched-test projection of classifier decision surfaces")
    _stamp_context(figure, context_label, bottom=0.12)
    return publish_figure(figure, artifacts, relative_path)


def ensemble_evidence_plot(
    forest_values: Mapping[str, tuple[float, float]],
    boosting_train: Sequence[float],
    boosting_validation: Sequence[float],
    *,
    score_label: str,
    context_label: str,
    artifacts: ArtifactStore,
    relative_path: str,
) -> str:
    """Contrast OOB/holdout evidence and stagewise boosting losses."""

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    names = tuple(forest_values)
    if names:
        positions = np.arange(len(names))
        width = 0.36
        axes[0].bar(
            positions - width / 2,
            [forest_values[name][0] for name in names],
            width,
            label="out-of-bag",
        )
        axes[0].bar(
            positions + width / 2,
            [forest_values[name][1] for name in names],
            width,
            label="holdout",
        )
        axes[0].set(xticks=positions, xticklabels=[name.replace("_", " ") for name in names])
        axes[0].legend(fontsize=8)
    axes[0].set(title="Forest generalization evidence", ylabel=score_label)
    axes[0].grid(axis="y", alpha=0.22)
    if boosting_train:
        axes[1].plot(np.arange(1, len(boosting_train) + 1), boosting_train, label="training")
    if boosting_validation:
        axes[1].plot(
            np.arange(1, len(boosting_validation) + 1),
            boosting_validation,
            label="validation",
        )
    axes[1].set(title="Boosting stage loss", xlabel="Stage", ylabel="Loss")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)
    _stamp_context(figure, context_label, bottom=0.13)
    return publish_figure(figure, artifacts, relative_path)
