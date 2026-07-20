"""Publication, validation, and resource-safety contracts for deep plots."""

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.reporting.deep import (
    DecisionPanel,
    accuracy_by_length_plot,
    confusion_matrix_plot,
    conv_filters_plot,
    decision_boundary_grid,
    error_histogram_plot,
    image_panel_plot,
    latent_scatter_plot,
    roc_curves_plot,
    scratch_loss_curves_plot,
    training_curves_plot,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_matplotlib_figures() -> Iterator[None]:
    plt.close("all")
    yield
    plt.close("all")


def _decision_panel() -> DecisionPanel:
    grid_x, grid_y = np.meshgrid(np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 4))
    return DecisionPanel(
        features=np.asarray([[-0.8, -0.5], [0.7, 0.6]], dtype=np.float64),
        targets=np.asarray([0, 1], dtype=np.int64),
        grid_x=np.asarray(grid_x, dtype=np.float64),
        grid_y=np.asarray(grid_y, dtype=np.float64),
        positive_probability=np.asarray((grid_x + 1.0) / 2.0, dtype=np.float64),
    )


def _assert_png(store: ArtifactStore, relative_path: str) -> None:
    destination = store.resolve(relative_path)
    assert destination.is_file()
    assert destination.stat().st_size > 100
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_all_deep_plots_publish_complete_pngs_and_close_figures(tmp_path: Path) -> None:
    plt.close("all")
    store = ArtifactStore(tmp_path)
    context = "seed=42 | train n=24 | validation n=8 | untouched test n=8 | cpu"

    published = [
        scratch_loss_curves_plot(
            {"xor": [0.8, 0.4], "moons": [0.9, 0.5]},
            context=context,
            artifacts=store,
            relative_path="scratch.png",
        ),
        decision_boundary_grid(
            {"xor": _decision_panel()},
            context=context,
            artifacts=store,
            relative_path="boundary.png",
        ),
        training_curves_plot(
            {"cnn": {"train_loss": [0.8, 0.4], "validation_loss": [0.9, 0.5]}},
            ylabel="cross-entropy",
            title="Training diagnostics",
            context=context,
            artifacts=store,
            relative_path="training.png",
            log_scale=True,
        ),
        confusion_matrix_plot(
            np.asarray([[3, 1], [0, 4]], dtype=np.int64),
            ["negative", "positive"],
            title="Held-out confusion matrix",
            context=context,
            artifacts=store,
            relative_path="confusion.png",
        ),
        image_panel_plot(
            {
                "original": np.zeros((2, 4, 4), dtype=np.float64),
                "reconstruction": np.ones((1, 4, 4), dtype=np.float64),
            },
            {"original": ["a", "b"], "reconstruction": ["a-hat"]},
            title="Reconstructions",
            context=context,
            artifacts=store,
            relative_path="images.png",
        ),
        conv_filters_plot(
            np.linspace(-1.0, 1.0, 18, dtype=np.float64).reshape(2, 1, 3, 3),
            context=context,
            artifacts=store,
            relative_path="filters.png",
        ),
        accuracy_by_length_plot(
            ["short", "long"],
            {"lstm": [0.8, 0.9], "padded_mlp": [0.7, 0.55]},
            context=context,
            artifacts=store,
            relative_path="length.png",
        ),
        error_histogram_plot(
            np.asarray([0.01, 0.02, 0.03]),
            np.asarray([0.2, 0.3, 0.4]),
            context=context,
            artifacts=store,
            relative_path="errors.png",
        ),
        roc_curves_plot(
            {
                "autoencoder": (
                    np.asarray([0.0, 0.0, 1.0]),
                    np.asarray([0.0, 1.0, 1.0]),
                    1.0,
                )
            },
            context=context,
            artifacts=store,
            relative_path="roc.png",
        ),
        latent_scatter_plot(
            np.asarray([[-1.0, 0.0], [1.0, 0.5], [0.0, -0.5]]),
            np.asarray([0, 1, 2], dtype=np.int64),
            axis_label="latent",
            context=context,
            artifacts=store,
            relative_path="latent.png",
        ),
    ]

    assert len(set(published)) == 10
    for relative_path in published:
        _assert_png(store, relative_path)
    assert {path.as_posix() for path in store.files()} == set(published)
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "call",
    [
        lambda store: scratch_loss_curves_plot(
            {}, context="ctx", artifacts=store, relative_path="unused.png"
        ),
        lambda store: decision_boundary_grid(
            {}, context="ctx", artifacts=store, relative_path="unused.png"
        ),
        lambda store: training_curves_plot(
            {},
            ylabel="loss",
            title="title",
            context="ctx",
            artifacts=store,
            relative_path="unused.png",
        ),
        lambda store: image_panel_plot(
            {},
            {},
            title="title",
            context="ctx",
            artifacts=store,
            relative_path="unused.png",
        ),
        lambda store: accuracy_by_length_plot(
            ["short"], {}, context="ctx", artifacts=store, relative_path="unused.png"
        ),
        lambda store: roc_curves_plot(
            {}, context="ctx", artifacts=store, relative_path="unused.png"
        ),
    ],
)
def test_mapping_plots_reject_empty_inputs_without_publication(
    tmp_path: Path, call: Callable[[ArtifactStore], str]
) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match=r"at least one|requires"):
        call(store)
    assert store.files() == ()


def test_plot_context_validation_does_not_leak_a_figure_or_partial_file(tmp_path: Path) -> None:
    plt.close("all")
    store = ArtifactStore(tmp_path)
    try:
        with pytest.raises(ValueError, match="context"):
            scratch_loss_curves_plot(
                {"model": [1.0]},
                context="   ",
                artifacts=store,
                relative_path="invalid.png",
            )
        assert store.files() == ()
        assert plt.get_fignums() == []
    finally:
        plt.close("all")


def test_artifact_path_failure_closes_the_figure_and_cannot_escape(tmp_path: Path) -> None:
    plt.close("all")
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="within"):
        scratch_loss_curves_plot(
            {"model": [1.0]},
            context="ctx",
            artifacts=store,
            relative_path="../escape.png",
        )

    assert store.files() == ()
    assert not (tmp_path.parent / "escape.png").exists()
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ([], "history|point|non-empty"),
        ([1.0, np.nan], "finite"),
    ],
)
def test_loss_curve_rejects_empty_or_nonfinite_series(
    tmp_path: Path, history: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        scratch_loss_curves_plot(
            {"model": history},
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


def test_decision_boundary_rejects_misaligned_panel_shapes(tmp_path: Path) -> None:
    panel = _decision_panel()
    invalid = DecisionPanel(
        features=np.zeros((2, 3)),
        targets=panel.targets,
        grid_x=panel.grid_x,
        grid_y=panel.grid_y,
        positive_probability=panel.positive_probability,
    )
    with pytest.raises(ValueError, match=r"feature|two"):
        decision_boundary_grid(
            {"bad": invalid},
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


def test_training_curve_rejects_candidate_without_series(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no series"):
        training_curves_plot(
            {"empty": {}},
            ylabel="loss",
            title="title",
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


@pytest.mark.parametrize(
    ("matrix", "labels", "message"),
    [
        (np.zeros((2, 3), dtype=np.int64), ["a", "b"], "shape"),
        (np.asarray([[1, -1], [0, 1]], dtype=np.int64), ["a", "b"], "non-negative"),
    ],
)
def test_confusion_plot_rejects_invalid_counts(
    tmp_path: Path, matrix: np.ndarray, labels: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        confusion_matrix_plot(
            matrix,
            labels,
            title="title",
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


@pytest.mark.parametrize(
    ("panels", "captions", "message"),
    [
        ({"empty": np.empty((0, 4, 4))}, {"empty": []}, "at least one image"),
        ({"row": np.zeros((2, 4, 4))}, {"row": ["only one"]}, "captions"),
        ({"row": np.zeros((2, 4, 3))}, {"row": ["a", "b"]}, "square|shape"),
    ],
)
def test_image_panel_rejects_invalid_rows(
    tmp_path: Path,
    panels: dict[str, np.ndarray],
    captions: dict[str, list[str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        image_panel_plot(
            panels,
            captions,
            title="title",
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        (np.zeros((2, 3, 3)), "single-channel|shaped"),
        (np.empty((0, 1, 3, 3)), "at least one|non-empty"),
        (np.full((1, 1, 3, 3), np.nan), "finite"),
    ],
)
def test_filter_plot_rejects_invalid_kernels(
    tmp_path: Path, filters: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        conv_filters_plot(
            filters,
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


@pytest.mark.parametrize(
    ("bucket_labels", "accuracies", "message"),
    [
        (["short", "long"], {"model": [0.5]}, "one accuracy"),
        (["short"], {"model": [1.1]}, r"\[0, 1\]|accuracy"),
        ([], {"model": []}, "bucket|non-empty|at least one"),
    ],
)
def test_accuracy_by_length_rejects_invalid_buckets_or_values(
    tmp_path: Path,
    bucket_labels: list[str],
    accuracies: dict[str, list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        accuracy_by_length_plot(
            bucket_labels,
            accuracies,
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


@pytest.mark.parametrize(
    ("inlier", "anomaly", "message"),
    [
        (np.asarray([]), np.asarray([1.0]), "non-empty"),
        (np.asarray([[0.1]]), np.asarray([1.0]), "1D|one-dimensional"),
        (np.asarray([np.nan]), np.asarray([1.0]), "finite"),
    ],
)
def test_error_histogram_rejects_invalid_error_samples(
    tmp_path: Path, inlier: np.ndarray, anomaly: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        error_histogram_plot(
            inlier,
            anomaly,
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


@pytest.mark.parametrize(
    ("fpr", "tpr", "auroc", "message"),
    [
        (np.asarray([0.0, 1.0]), np.asarray([0.0]), 0.5, "shape|length"),
        (np.asarray([0.0, 0.8, 0.7, 1.0]), np.asarray([0.0, 0.5, 0.8, 1.0]), 0.7, "monotonic"),
        (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), 1.1, r"\[0, 1\]|AUROC"),
    ],
)
def test_roc_plot_rejects_invalid_curve_contracts(
    tmp_path: Path,
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        roc_curves_plot(
            {"model": (fpr, tpr, auroc)},
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )


@pytest.mark.parametrize(
    ("latent", "labels", "message"),
    [
        (np.zeros((3, 3)), np.asarray([0, 1, 2]), r"\(n, 2\)"),
        (np.zeros((3, 2)), np.asarray([0, 1]), "one label"),
        (np.full((3, 2), np.nan), np.asarray([0, 1, 2]), "finite"),
        (np.zeros((3, 2)), np.asarray([[0], [1], [2]]), "label|1D"),
    ],
)
def test_latent_plot_rejects_invalid_coordinates_or_labels(
    tmp_path: Path, latent: np.ndarray, labels: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        latent_scatter_plot(
            latent,
            labels,
            axis_label="latent",
            context="ctx",
            artifacts=ArtifactStore(tmp_path),
            relative_path="invalid.png",
        )
