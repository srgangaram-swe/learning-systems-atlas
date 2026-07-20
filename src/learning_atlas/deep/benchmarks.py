"""Deep-learning benchmark experiments with reproducible, committed evidence.

Four studies cover the Sprint 4 milestone: the autograd MLP on linearly
inseparable planar tasks, the CNN-versus-MLP digits comparison, the LSTM
long-range sequence study, and the bottleneck-autoencoder anomaly study. Every
study derives named seed streams, trains deterministically on CPU, and
publishes plots, records, and model state through the transactional runner.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import JsonValue

from learning_atlas.core.config import (
    DeepAutoencoderBenchmarkConfig,
    DeepSequenceBenchmarkConfig,
    DeepVisionBenchmarkConfig,
    ScratchMLPBenchmarkConfig,
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
from learning_atlas.core.reproducibility import derive_named_seed, generator_for_seed
from learning_atlas.core.validation import FloatArray
from learning_atlas.deep.autoencoder import BottleneckAutoencoder, reconstruction_errors
from learning_atlas.deep.autograd import finite_difference_gradient
from learning_atlas.deep.datasets import (
    PlanarDataset,
    make_temporal_xor,
    make_two_moons,
    make_xor,
)
from learning_atlas.deep.mlp import MLPClassifier
from learning_atlas.deep.sequence import PaddedSequenceMLP, SequenceLSTM
from learning_atlas.deep.training import (
    ClassificationObjective,
    Objective,
    ReconstructionObjective,
    SequenceClassificationObjective,
    Trainer,
    TrainerConfig,
    TrainingReport,
    seed_torch,
    tensor_dataset,
)
from learning_atlas.deep.vision import CompactCNN, VisionMLP
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
from learning_atlas.supervised.linear_model import LogisticRegression
from learning_atlas.supervised.metrics import (
    binary_roc_curve,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from learning_atlas.unsupervised.decomposition import PCA
from learning_atlas.unsupervised.metrics import silhouette_analysis

IntArray = NDArray[np.int64]
_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True, slots=True)
class _CandidateRecord:
    study: str
    candidate: str
    seed: int | None
    parameters: dict[str, JsonValue]
    metrics: dict[str, float]
    named_seeds: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class _ScratchTaskFit:
    task: PlanarDataset
    train_indices: IntArray
    validation_indices: IntArray
    test_indices: IntArray
    mlp: MLPClassifier
    baseline: LogisticRegression
    candidate_seeds: dict[str, int]
    validation_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class _TorchCandidateFit:
    name: str
    seed: int
    model: torch.nn.Module
    trainer: Trainer
    report: TrainingReport
    validation_loss: float
    validation_metrics: dict[str, float]


def _write_candidate_records(
    context: RunContext, records: tuple[_CandidateRecord, ...]
) -> tuple[str, str]:
    json_path = "records/candidate_records.json"
    csv_path = "records/candidate_records.csv"
    payload = [
        {
            "study": record.study,
            "candidate": record.candidate,
            "seed": record.seed,
            "named_seeds": record.named_seeds,
            "parameters": record.parameters,
            "metrics": record.metrics,
        }
        for record in records
    ]
    context.artifacts.write_json(json_path, payload)

    metric_columns = sorted({key for record in records for key in record.metrics})
    with context.artifacts.atomic_target(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["study", "candidate", "seed", "named_seeds", *metric_columns],
                extrasaction="raise",
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "study": record.study,
                        "candidate": record.candidate,
                        "seed": record.seed,
                        "named_seeds": ";".join(
                            f"{name}={seed}"
                            for name, seed in sorted((record.named_seeds or {}).items())
                        ),
                        **record.metrics,
                    }
                )
    return json_path, csv_path


def _three_way_indices(
    targets: IntArray,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    stratify: bool,
) -> tuple[IntArray, IntArray, IntArray]:
    """Split sample indices deterministically, optionally class-proportionally."""

    target_array = np.asarray(targets)
    if target_array.ndim != 1 or len(target_array) < 3:
        msg = "three-way split requires a one-dimensional target array with at least 3 rows"
        raise ValueError(msg)
    for name, fraction in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            msg = f"{name} must be a finite value in (0, 1)"
            raise ValueError(msg)
    if validation_fraction + test_fraction >= 1.0:
        msg = "validation_fraction plus test_fraction must be below 1"
        raise ValueError(msg)
    rng = generator_for_seed(seed)
    groups = (
        [np.flatnonzero(target_array == label) for label in np.unique(target_array)]
        if stratify
        else [np.arange(len(target_array), dtype=np.int64)]
    )
    train_parts: list[IntArray] = []
    validation_parts: list[IntArray] = []
    test_parts: list[IntArray] = []
    for group in groups:
        shuffled = rng.permutation(group)
        n_test = round(test_fraction * len(shuffled))
        n_validation = round(validation_fraction * len(shuffled))
        test_parts.append(np.asarray(shuffled[:n_test], dtype=np.int64))
        validation_parts.append(
            np.asarray(shuffled[n_test : n_test + n_validation], dtype=np.int64)
        )
        train_parts.append(np.asarray(shuffled[n_test + n_validation :], dtype=np.int64))
    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    test = np.concatenate(test_parts)
    if min(len(train), len(validation), len(test)) == 0:
        msg = "three-way split produced an empty partition; enlarge the dataset"
        raise ValueError(msg)
    return np.sort(train), np.sort(validation), np.sort(test)


def _loss_curves(report: TrainingReport) -> dict[str, list[float]]:
    return {
        "train_loss": [record.train_loss for record in report.history],
        "validation_loss": [record.validation_loss for record in report.history],
    }


def _accuracy_curve(report: TrainingReport) -> dict[str, list[float]]:
    return {"validation_accuracy": [float(record.metrics["accuracy"]) for record in report.history]}


def _predict_logits(model: torch.nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            logits: torch.Tensor = model(*(value.to(device) for value in inputs))
    finally:
        model.train(was_training)
    return logits.detach().cpu()


def _encode_latent(model: BottleneckAutoencoder, inputs: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            latent = model.encode(inputs.to(device))
    finally:
        model.train(was_training)
    return latent.detach().cpu()


def _save_state_dict(context: RunContext, model: torch.nn.Module, relative_path: str) -> str:
    with context.artifacts.atomic_target(relative_path) as temporary:
        serialized = BytesIO()
        torch.save(
            {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            serialized,
        )
        temporary.write_bytes(serialized.getvalue())
    return relative_path


def _preserve_torch_state(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Restore caller-owned RNG and deterministic-kernel policy around a run."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        rng_state = torch.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cuda_initialized = torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else None
        try:
            return function(*args, **kwargs)
        finally:
            torch.set_rng_state(rng_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    return wrapped


def _select_candidate(candidates: tuple[CandidateResult, ...], *, metric: str, prefer: str) -> str:
    """Pick by an isolated validation metric with a deterministic preferred tie."""

    if not candidates:
        msg = "candidate selection requires at least one candidate"
        raise ValueError(msg)
    if prefer not in {candidate.name for candidate in candidates}:
        msg = f"preferred candidate {prefer!r} is not present"
        raise ValueError(msg)
    declared_fields = {metric}
    for candidate in candidates:
        observed_fields = set(candidate.metrics)
        if observed_fields != declared_fields:
            msg = (
                f"candidate {candidate.name!r} must expose only selection metric "
                f"{metric!r}; received {sorted(observed_fields)!r}"
            )
            raise ValueError(msg)
        if not math.isfinite(candidate.metrics[metric]):
            msg = f"candidate {candidate.name!r} has a non-finite selection metric"
            raise ValueError(msg)

    return max(
        candidates,
        key=lambda candidate: (candidate.metrics[metric], candidate.name == prefer),
    ).name


# --------------------------------------------------------------------------- #24/#25


class ScratchMLPBenchmark:
    """Autograd MLP versus a linear baseline on tasks that require depth."""

    def __init__(self, config: ScratchMLPBenchmarkConfig) -> None:
        self._config = config

    def _fit_task(self, task: PlanarDataset, context: RunContext) -> _ScratchTaskFit:
        """Fit both candidates and expose training/validation evidence only."""

        config = self._config
        split_seed = derive_named_seed(context.seed, config.experiment, "split", task.name)
        train_idx, validation_idx, test_idx = _three_way_indices(
            task.targets,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_size,
            seed=split_seed,
            stratify=True,
        )
        mlp_seed = derive_named_seed(
            context.seed, config.experiment, "candidate", "autograd_mlp", task.name
        )
        mlp = MLPClassifier(
            (config.hidden_units,),
            activation=config.activation,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            seed=mlp_seed,
        ).fit(task.features[train_idx], task.targets[train_idx])
        baseline_seed = derive_named_seed(
            context.seed, config.experiment, "candidate", "logistic_baseline", task.name
        )
        baseline = LogisticRegression(seed=baseline_seed).fit(
            task.features[train_idx], task.targets[train_idx]
        )

        validation_metrics: dict[str, dict[str, float]] = {}
        for name, model in (("autograd_mlp", mlp), ("logistic_baseline", baseline)):
            validation_metrics[name] = {
                f"{task.name}_train_accuracy": model.score(
                    task.features[train_idx], task.targets[train_idx]
                ),
                f"{task.name}_validation_accuracy": model.score(
                    task.features[validation_idx], task.targets[validation_idx]
                ),
                f"{task.name}_validation_log_loss": log_loss(
                    task.targets[validation_idx],
                    model.predict_proba(task.features[validation_idx]),
                ),
            }
        return _ScratchTaskFit(
            task=task,
            train_indices=train_idx,
            validation_indices=validation_idx,
            test_indices=test_idx,
            mlp=mlp,
            baseline=baseline,
            candidate_seeds={
                "autograd_mlp": mlp_seed,
                "logistic_baseline": baseline_seed,
            },
            validation_metrics=validation_metrics,
        )

    @staticmethod
    def _gradient_check(mlp: MLPClassifier, features: FloatArray, targets: FloatArray) -> float:
        loss = mlp.batch_loss(features, targets)
        for parameter in mlp.parameters:
            parameter.zero_grad()
        loss.backward()
        worst = 0.0
        for parameter in mlp.parameters:
            analytic = parameter.grad
            assert analytic is not None  # backward reaches every layer parameter
            numeric = finite_difference_gradient(
                lambda: mlp.batch_loss(features, targets), parameter
            )
            worst = max(worst, float(np.max(np.abs(numeric - analytic))))
        return worst

    @staticmethod
    def _probability_panel(mlp: MLPClassifier, task: PlanarDataset) -> DecisionPanel:
        margin = 0.6
        xs = np.linspace(
            task.features[:, 0].min() - margin, task.features[:, 0].max() + margin, 160
        )
        ys = np.linspace(
            task.features[:, 1].min() - margin, task.features[:, 1].max() + margin, 160
        )
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        positive_column = int(np.searchsorted(mlp.classes_, 1.0))
        probabilities = mlp.predict_proba(points)[:, positive_column]
        return DecisionPanel(
            features=task.features,
            targets=task.targets,
            grid_x=grid_x,
            grid_y=grid_y,
            positive_probability=probabilities.reshape(grid_x.shape),
        )

    def run(self, context: RunContext) -> RunResult:
        config = self._config
        tasks = (
            make_xor(
                config.n_samples,
                noise=config.xor_noise,
                seed=derive_named_seed(context.seed, config.experiment, "data", "xor"),
            ),
            make_two_moons(
                config.n_samples,
                noise=config.moons_noise,
                seed=derive_named_seed(context.seed, config.experiment, "data", "two_moons"),
            ),
        )

        fits = tuple(self._fit_task(task, context) for task in tasks)
        selection_candidates = tuple(
            CandidateResult(
                name=name,
                metrics={
                    "mean_validation_accuracy": float(
                        np.mean(
                            [
                                fit.validation_metrics[name][f"{fit.task.name}_validation_accuracy"]
                                for fit in fits
                            ]
                        )
                    )
                },
            )
            for name in ("autograd_mlp", "logistic_baseline")
        )
        selected = _select_candidate(
            selection_candidates,
            metric="mean_validation_accuracy",
            prefer="autograd_mlp",
        )

        # The held-out partitions and diagnostics are opened only after selection.
        merged: dict[str, dict[str, float]] = {"autograd_mlp": {}, "logistic_baseline": {}}
        panels: dict[str, DecisionPanel] = {}
        histories: dict[str, list[float]] = {}
        weights_payload: dict[str, FloatArray] = {}
        for fit in fits:
            task = fit.task
            for candidate, values in fit.validation_metrics.items():
                merged[candidate].update(values)
            for name, model in (
                ("autograd_mlp", fit.mlp),
                ("logistic_baseline", fit.baseline),
            ):
                merged[name][f"{task.name}_test_accuracy"] = model.score(
                    task.features[fit.test_indices], task.targets[fit.test_indices]
                )
                merged[name][f"{task.name}_test_log_loss"] = log_loss(
                    task.targets[fit.test_indices],
                    model.predict_proba(task.features[fit.test_indices]),
                )
            frozen = min(config.gradient_check_samples, len(fit.train_indices))
            merged["autograd_mlp"][f"{task.name}_gradient_check_max_abs_error"] = (
                self._gradient_check(
                    fit.mlp,
                    task.features[fit.train_indices][:frozen],
                    np.asarray(task.targets[fit.train_indices][:frozen], dtype=np.float64),
                )
            )
            merged["autograd_mlp"][f"{task.name}_final_training_loss"] = fit.mlp.loss_history_[-1]
            panels[task.name] = self._probability_panel(fit.mlp, task)
            histories[task.name] = list(fit.mlp.loss_history_)
            for index, parameter in enumerate(fit.mlp.parameters):
                weights_payload[f"{task.name}_parameter_{index}"] = parameter.data
        selection_by_name = {candidate.name: candidate for candidate in selection_candidates}
        for name, values in merged.items():
            values["mean_validation_accuracy"] = selection_by_name[name].metrics[
                "mean_validation_accuracy"
            ]
            values["mean_test_accuracy"] = float(
                np.mean([values[f"{task.name}_test_accuracy"] for task in tasks])
            )
        gradient_errors = [
            merged["autograd_mlp"][f"{task.name}_gradient_check_max_abs_error"] for task in tasks
        ]
        merged["autograd_mlp"]["gradient_check_max_abs_error"] = float(np.max(gradient_errors))

        candidates = tuple(
            CandidateResult(name=name, metrics=metrics) for name, metrics in merged.items()
        )

        records = tuple(
            _CandidateRecord(
                study="planar_tasks",
                candidate=name,
                seed=None,
                named_seeds={fit.task.name: fit.candidate_seeds[name] for fit in fits},
                parameters={
                    "hidden_units": config.hidden_units,
                    "activation": config.activation,
                    "learning_rate": config.learning_rate,
                    "momentum": config.momentum,
                    "batch_size": config.batch_size,
                    "max_epochs": config.max_epochs,
                }
                if name == "autograd_mlp"
                else {
                    "model": "l2_logistic_regression",
                    "solver": "gradient_descent",
                },
                metrics=metrics,
            )
            for name, metrics in merged.items()
        )
        records_json, records_csv = _write_candidate_records(context, records)

        weights_path = "models/autograd_mlp_parameters.npz"
        with context.artifacts.atomic_target(weights_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **weights_payload)  # type: ignore[arg-type]

        plot_context = (
            f"seed {context.seed} | {config.n_samples} samples/task | "
            f"hidden {config.hidden_units} {config.activation} | "
            f"SGD lr {config.learning_rate} momentum {config.momentum}"
        )
        loss_plot = scratch_loss_curves_plot(
            histories,
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/mlp_loss_curves.png",
        )
        boundary_plot = decision_boundary_grid(
            panels,
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/mlp_decision_boundaries.png",
        )

        source = SourceMetadata(
            kind=SourceKind.GENERATOR,
            name="planar_tasks",
            version="1.0.0",
            fingerprint_sha256=array_fingerprint(
                *(task.features for task in tasks), *(task.targets for task in tasks)
            ),
            target_used_for_fit=True,
            details={
                "tasks": [task.name for task in tasks],
                "n_samples_per_task": config.n_samples,
                "validation_fraction": config.validation_fraction,
                "test_fraction": config.test_size,
                "xor_noise": config.xor_noise,
                "moons_noise": config.moons_noise,
            },
        )
        selected_result = next(candidate for candidate in candidates if candidate.name == selected)
        return RunResult(
            experiment=config.experiment,
            paradigm=LearningParadigm.DEEP,
            seed=context.seed,
            source=source,
            selected_model=selected,
            metrics=dict(selected_result.metrics),
            candidates=candidates,
            artifacts={
                "candidate_records_json": records_json,
                "candidate_records_csv": records_csv,
                "mlp_parameters": weights_path,
                "loss_curves_plot": loss_plot,
                "decision_boundary_plot": boundary_plot,
            },
            notes=(
                "Every gradient flows through the from-scratch reverse-mode engine; "
                "finite-difference checks on a frozen minibatch bound the analytic error.",
                "The linear baseline documents why these tasks need hidden units.",
                "Candidate choice uses validation accuracy; untouched-test metrics are "
                "opened only after both fits are frozen.",
            ),
        )


# ------------------------------------------------------------------------------- #27


def _load_digit_images(
    n_samples: int,
    *,
    seed: int,
    sampling: Literal["label_stratified", "uniform"] = "label_stratified",
) -> tuple[FloatArray, IntArray]:
    """Load digits using either label-stratified or label-independent sampling."""

    from sklearn.datasets import load_digits

    bundle = load_digits()
    images = np.asarray(bundle.images, dtype=np.float64) / 16.0
    targets = np.asarray(bundle.target, dtype=np.int64)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        msg = "n_samples must be a positive integer"
        raise ValueError(msg)
    if n_samples > len(targets):
        msg = f"requested {n_samples} digits but only {len(targets)} exist"
        raise ValueError(msg)
    if sampling not in {"label_stratified", "uniform"}:
        msg = "sampling must be 'label_stratified' or 'uniform'"
        raise ValueError(msg)
    rng = generator_for_seed(seed)
    if sampling == "uniform":
        keep = np.sort(
            np.asarray(rng.choice(len(images), size=n_samples, replace=False), dtype=np.int64)
        )
    else:
        labels, counts = np.unique(targets, return_counts=True)
        expected = n_samples * counts.astype(np.float64) / len(targets)
        quotas = np.floor(expected).astype(np.int64)
        remaining = n_samples - int(np.sum(quotas))
        allocation_order = np.lexsort((labels, -(expected - quotas)))
        for position in allocation_order[:remaining]:
            quotas[position] += 1
        keep_parts: list[IntArray] = []
        for label, quota in zip(labels, quotas, strict=True):
            label_indices = rng.permutation(np.flatnonzero(targets == label))
            keep_parts.append(np.asarray(label_indices[:quota], dtype=np.int64))
        keep = np.sort(np.concatenate(keep_parts))
    if len(keep) != n_samples:  # defensive guard around the allocation invariant
        msg = f"digit sampling produced {len(keep)} rows instead of {n_samples}"
        raise RuntimeError(msg)
    selected_images = np.asarray(images[keep], dtype=np.float64)
    selected_targets = np.asarray(targets[keep], dtype=np.int64)
    selected_images.setflags(write=False)
    selected_targets.setflags(write=False)
    return selected_images, selected_targets


class DeepVisionBenchmark:
    """A compact CNN against a dense baseline on 8x8 digits through one Trainer."""

    def __init__(self, config: DeepVisionBenchmarkConfig) -> None:
        self._config = config

    def _trainer(self, name: str, seed: int) -> Trainer:
        config = self._config
        return Trainer(
            TrainerConfig(
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=derive_named_seed(seed, config.experiment, "trainer", name),
                patience=config.patience,
                min_delta=1e-4,
                device=config.device,
            )
        )

    def _build_model(self, name: str) -> torch.nn.Module:
        if name == "compact_cnn":
            return CompactCNN(
                image_size=8,
                channels=self._config.channels,
                hidden_units=self._config.hidden_units,
                n_classes=10,
            )
        return VisionMLP(image_size=8, hidden_units=self._config.hidden_units, n_classes=10)

    @_preserve_torch_state
    def run(self, context: RunContext) -> RunResult:
        config = self._config
        images, targets = _load_digit_images(
            config.n_samples,
            seed=derive_named_seed(context.seed, config.experiment, "data"),
            sampling="label_stratified",
        )
        train_idx, validation_idx, test_idx = _three_way_indices(
            targets,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=derive_named_seed(context.seed, config.experiment, "split"),
            stratify=True,
        )
        inputs = torch.from_numpy(images[:, np.newaxis, :, :].astype(np.float32))
        labels = torch.from_numpy(np.array(targets, dtype=np.int64, copy=True))
        development_datasets = {
            "train": tensor_dataset(inputs[train_idx], labels[train_idx]),
            "validation": tensor_dataset(inputs[validation_idx], labels[validation_idx]),
        }
        objective = ClassificationObjective()

        fitted: list[_TorchCandidateFit] = []
        artifacts: dict[str, str] = {}
        for name in ("vision_mlp", "compact_cnn"):
            model_seed = derive_named_seed(context.seed, config.experiment, "candidate", name)
            seed_torch(model_seed)
            model = self._build_model(name)
            trainer = self._trainer(name, context.seed)
            checkpoint_relative = f"models/{name}_checkpoint.pt"
            checkpoint_path = context.artifacts.resolve(checkpoint_relative)
            report = trainer.fit(
                model,
                development_datasets["train"],
                development_datasets["validation"],
                objective,
                checkpoint_path=checkpoint_path,
            )
            validation_loss, validation_metrics = trainer.evaluate(
                model, development_datasets["validation"], objective
            )
            fitted.append(
                _TorchCandidateFit(
                    name=name,
                    seed=model_seed,
                    model=model,
                    trainer=trainer,
                    report=report,
                    validation_loss=validation_loss,
                    validation_metrics=validation_metrics,
                )
            )
            artifacts[f"{name}_checkpoint"] = checkpoint_relative

        selection_candidates = tuple(
            CandidateResult(
                name=fit.name,
                metrics={"selection_validation_accuracy": fit.validation_metrics["accuracy"]},
            )
            for fit in fitted
        )
        selected = _select_candidate(
            selection_candidates,
            metric="selection_validation_accuracy",
            prefer="compact_cnn",
        )

        # Held-out evaluation starts only after the validation-only choice is frozen.
        test_dataset = tensor_dataset(inputs[test_idx], labels[test_idx])
        merged: dict[str, dict[str, float]] = {}
        curves: dict[str, dict[str, list[float]]] = {}
        accuracy_curves: dict[str, dict[str, list[float]]] = {}
        models: dict[str, torch.nn.Module] = {}
        records: list[_CandidateRecord] = []
        for fit in fitted:
            test_loss, test_metrics = fit.trainer.evaluate(fit.model, test_dataset, objective)
            predictions = (
                _predict_logits(fit.model, inputs[test_idx]).argmax(dim=1).numpy().astype(np.int64)
            )
            reload_deltas = self._checkpoint_reload_deltas(
                fit.name,
                fit.model,
                context.artifacts.resolve(f"models/{fit.name}_checkpoint.pt"),
                fit.trainer,
                test_dataset,
                objective,
                reference_loss=test_loss,
                reference_metrics=test_metrics,
            )
            merged[fit.name] = {
                "selection_validation_accuracy": fit.validation_metrics["accuracy"],
                "selection_validation_loss": fit.validation_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": f1_score(targets[test_idx], predictions),
                "test_loss": test_loss,
                "best_epoch": float(fit.report.best_epoch),
                "epochs_run": float(len(fit.report.history)),
                "stopped_early": float(fit.report.stopped_early),
                **reload_deltas,
                "parameter_count": float(
                    sum(parameter.numel() for parameter in fit.model.parameters())
                ),
            }
            curves[fit.name] = _loss_curves(fit.report)
            accuracy_curves[fit.name] = _accuracy_curve(fit.report)
            models[fit.name] = fit.model
            records.append(
                _CandidateRecord(
                    study="digits_8x8",
                    candidate=fit.name,
                    seed=fit.seed,
                    parameters={
                        "max_epochs": config.max_epochs,
                        "batch_size": config.batch_size,
                        "learning_rate": config.learning_rate,
                        "patience": config.patience,
                        "hidden_units": config.hidden_units,
                        "channels": config.channels if fit.name == "compact_cnn" else None,
                        "optimizer": "adam",
                    },
                    metrics=merged[fit.name],
                )
            )
        merged["compact_cnn"]["accuracy_gain_over_mlp"] = (
            merged["compact_cnn"]["test_accuracy"] - merged["vision_mlp"]["test_accuracy"]
        )

        candidates = tuple(
            CandidateResult(name=name, metrics=metrics) for name, metrics in merged.items()
        )
        records_json, records_csv = _write_candidate_records(context, tuple(records))

        cnn = models["compact_cnn"]
        assert isinstance(cnn, CompactCNN)
        cnn_predictions = (
            _predict_logits(cnn, inputs[test_idx]).argmax(dim=1).numpy().astype(np.int64)
        )
        plot_context = (
            f"seed {context.seed} | {config.n_samples} digits | "
            f"train/val/test {len(train_idx)}/{len(validation_idx)}/{len(test_idx)} | "
            f"Adam lr {config.learning_rate} | patience {config.patience}"
        )
        artifacts["training_loss_plot"] = training_curves_plot(
            curves,
            ylabel="cross-entropy",
            title="Digits training and validation loss",
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/vision_training_loss.png",
            log_scale=True,
        )
        artifacts["validation_accuracy_plot"] = training_curves_plot(
            accuracy_curves,
            ylabel="validation accuracy",
            title="Digits validation accuracy",
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/vision_validation_accuracy.png",
        )
        artifacts["confusion_matrix_plot"] = confusion_matrix_plot(
            confusion_matrix(targets[test_idx], cnn_predictions, labels=np.arange(10)),
            [str(digit) for digit in range(10)],
            title="Compact CNN held-out confusion",
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/vision_confusion_matrix.png",
        )
        wrong = np.flatnonzero(cnn_predictions != targets[test_idx])[:10]
        if len(wrong) > 0:
            artifacts["misclassified_plot"] = image_panel_plot(
                {"misclassified": images[test_idx][wrong]},
                {
                    "misclassified": [
                        f"true {targets[test_idx][index]} / pred {cnn_predictions[index]}"
                        for index in wrong
                    ]
                },
                title="Every remaining CNN mistake on the held-out digits",
                context=plot_context,
                artifacts=context.artifacts,
                relative_path="plots/vision_misclassified.png",
            )
        artifacts["conv_filters_plot"] = conv_filters_plot(
            np.asarray(cnn.first_layer_filters().numpy(), dtype=np.float64),
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/vision_conv_filters.png",
        )
        artifacts["candidate_records_json"] = records_json
        artifacts["candidate_records_csv"] = records_csv

        source = SourceMetadata(
            kind=SourceKind.DATASET,
            name="sklearn_digits_8x8",
            version=version("scikit-learn"),
            fingerprint_sha256=array_fingerprint(images, targets),
            target_used_for_fit=True,
            details={
                "n_samples_used": len(targets),
                "n_train": len(train_idx),
                "n_validation": len(validation_idx),
                "n_test": len(test_idx),
                "pixel_range": "[0, 1] after /16 normalization",
            },
        )
        selected_result = next(candidate for candidate in candidates if candidate.name == selected)
        return RunResult(
            experiment=config.experiment,
            paradigm=LearningParadigm.DEEP,
            seed=context.seed,
            source=source,
            selected_model=selected,
            metrics=dict(selected_result.metrics),
            candidates=candidates,
            artifacts=artifacts,
            notes=(
                "Both candidates share the Trainer, split, seed policy, and budget; "
                "only the inductive bias differs.",
                "Checkpoint reload is verified by re-evaluating restored best weights.",
            ),
        )

    def _checkpoint_reload_deltas(
        self,
        name: str,
        reference_model: torch.nn.Module,
        checkpoint_path: Path,
        trainer: Trainer,
        test_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, ...]],
        objective: Objective,
        *,
        reference_loss: float,
        reference_metrics: Mapping[str, float],
    ) -> dict[str, float]:
        """Restore best weights and compare state, logits, loss, and metrics."""

        payload = Trainer.load_checkpoint(checkpoint_path)
        fresh = self._build_model(name)
        fresh.load_state_dict(payload["best_model_state"])
        reloaded_loss, reloaded_metrics = trainer.evaluate(fresh, test_dataset, objective)
        reference_state = reference_model.state_dict()
        state_delta = max(
            float(torch.max(torch.abs(reference_state[key].cpu() - value)).item())
            for key, value in payload["best_model_state"].items()
        )
        reference_logits: list[torch.Tensor] = []
        reloaded_logits: list[torch.Tensor] = []
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=self._config.batch_size):
            reference_logits.append(_predict_logits(reference_model, batch[0]))
            reloaded_logits.append(_predict_logits(fresh, batch[0]))
        logit_delta = float(
            torch.max(torch.abs(torch.cat(reference_logits) - torch.cat(reloaded_logits))).item()
        )
        return {
            "checkpoint_reload_state_max_abs_delta": state_delta,
            "checkpoint_reload_logit_max_abs_delta": logit_delta,
            "checkpoint_reload_loss_delta": abs(reloaded_loss - reference_loss),
            "checkpoint_reload_accuracy_delta": abs(
                reloaded_metrics["accuracy"] - reference_metrics["accuracy"]
            ),
        }


# ------------------------------------------------------------------------------- #28


class DeepSequenceBenchmark:
    """An LSTM against a position-bound MLP on the variable-length temporal-XOR task."""

    def __init__(self, config: DeepSequenceBenchmarkConfig) -> None:
        self._config = config

    def _build_model(self, name: str) -> torch.nn.Module:
        if name == "sequence_lstm":
            return SequenceLSTM(input_size=1, hidden_size=self._config.hidden_size, n_classes=2)
        return PaddedSequenceMLP(
            max_length=self._config.max_length,
            input_size=1,
            hidden_units=self._config.mlp_hidden_units,
            n_classes=2,
        )

    def _trainer(self, name: str, seed: int) -> Trainer:
        config = self._config
        return Trainer(
            TrainerConfig(
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=derive_named_seed(seed, config.experiment, "trainer", name),
                patience=config.patience,
                min_delta=1e-5,
                device=config.device,
            )
        )

    @staticmethod
    def _padding_invariance_delta(
        model: SequenceLSTM, padded: torch.Tensor, lengths: torch.Tensor
    ) -> float:
        """Measure how much logits move when extra padding is appended."""

        probe = min(64, int(padded.shape[0]))
        base = _predict_logits(model, padded[:probe], lengths[:probe])
        extra = torch.cat([padded[:probe], torch.zeros(probe, 8, padded.shape[2])], dim=1)
        widened = _predict_logits(model, extra, lengths[:probe])
        return float(torch.max(torch.abs(base - widened)).item())

    @staticmethod
    def _length_buckets(
        lengths: IntArray, *, min_length: int, max_length: int
    ) -> tuple[tuple[str, ...], list[IntArray]]:
        edges = np.linspace(min_length, max_length + 1, 4).astype(np.int64)
        labels: list[str] = []
        buckets: list[IntArray] = []
        for lower, upper in pairwise(edges):
            member = np.flatnonzero((lengths >= lower) & (lengths < upper))
            if len(member) == 0:
                continue
            labels.append(f"{lower}-{upper - 1}")
            buckets.append(member)
        return tuple(labels), buckets

    @_preserve_torch_state
    def run(self, context: RunContext) -> RunResult:
        config = self._config
        dataset = make_temporal_xor(
            config.n_sequences,
            min_length=config.min_length,
            max_length=config.max_length,
            seed=derive_named_seed(context.seed, config.experiment, "data"),
        )
        train_idx, validation_idx, test_idx = _three_way_indices(
            dataset.targets,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=derive_named_seed(context.seed, config.experiment, "split"),
            stratify=True,
        )
        padded = torch.from_numpy(dataset.sequences.astype(np.float32))
        lengths = torch.from_numpy(np.array(dataset.lengths, dtype=np.int64, copy=True))
        labels = torch.from_numpy(np.array(dataset.targets, dtype=np.int64, copy=True))
        development_partitions = {
            "train": tensor_dataset(padded[train_idx], lengths[train_idx], labels[train_idx]),
            "validation": tensor_dataset(
                padded[validation_idx], lengths[validation_idx], labels[validation_idx]
            ),
        }
        objective = SequenceClassificationObjective()

        fitted: list[_TorchCandidateFit] = []
        for name in ("padded_mlp", "sequence_lstm"):
            model_seed = derive_named_seed(context.seed, config.experiment, "candidate", name)
            seed_torch(model_seed)
            model = self._build_model(name)
            trainer = self._trainer(name, context.seed)
            report = trainer.fit(
                model,
                development_partitions["train"],
                development_partitions["validation"],
                objective,
            )
            validation_loss, validation_metrics = trainer.evaluate(
                model, development_partitions["validation"], objective
            )
            fitted.append(
                _TorchCandidateFit(
                    name=name,
                    seed=model_seed,
                    model=model,
                    trainer=trainer,
                    report=report,
                    validation_loss=validation_loss,
                    validation_metrics=validation_metrics,
                )
            )

        selection_candidates = tuple(
            CandidateResult(
                name=fit.name,
                metrics={"selection_validation_accuracy": fit.validation_metrics["accuracy"]},
            )
            for fit in fitted
        )
        selected = _select_candidate(
            selection_candidates,
            metric="selection_validation_accuracy",
            prefer="sequence_lstm",
        )

        # Held-out predictions and length diagnostics begin after selection.
        test_partition = tensor_dataset(padded[test_idx], lengths[test_idx], labels[test_idx])
        bucket_labels, buckets = self._length_buckets(
            np.asarray(dataset.lengths[test_idx], dtype=np.int64),
            min_length=config.min_length,
            max_length=config.max_length,
        )
        merged: dict[str, dict[str, float]] = {}
        curves: dict[str, dict[str, list[float]]] = {}
        accuracy_curves: dict[str, dict[str, list[float]]] = {}
        bucket_accuracies: dict[str, list[float]] = {}
        records: list[_CandidateRecord] = []
        artifacts: dict[str, str] = {}
        models: dict[str, torch.nn.Module] = {}
        for fit in fitted:
            test_loss, test_metrics = fit.trainer.evaluate(fit.model, test_partition, objective)
            predictions = (
                _predict_logits(fit.model, padded[test_idx], lengths[test_idx])
                .argmax(dim=1)
                .numpy()
                .astype(np.int64)
            )
            observed = np.asarray(dataset.targets[test_idx], dtype=np.int64)
            bucket_accuracies[fit.name] = [
                float(np.mean(predictions[bucket] == observed[bucket])) for bucket in buckets
            ]
            merged[fit.name] = {
                "selection_validation_accuracy": fit.validation_metrics["accuracy"],
                "selection_validation_loss": fit.validation_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_loss,
                "best_epoch": float(fit.report.best_epoch),
                "epochs_run": float(len(fit.report.history)),
                "stopped_early": float(fit.report.stopped_early),
                "parameter_count": float(
                    sum(parameter.numel() for parameter in fit.model.parameters())
                ),
            }
            curves[fit.name] = _loss_curves(fit.report)
            accuracy_curves[fit.name] = _accuracy_curve(fit.report)
            models[fit.name] = fit.model
            state_path = f"models/{fit.name}_state.pt"
            artifacts[f"{fit.name}_state"] = _save_state_dict(context, fit.model, state_path)
            records.append(
                _CandidateRecord(
                    study="temporal_xor",
                    candidate=fit.name,
                    seed=fit.seed,
                    parameters={
                        "hidden_size": config.hidden_size
                        if fit.name == "sequence_lstm"
                        else config.mlp_hidden_units,
                        "max_epochs": config.max_epochs,
                        "batch_size": config.batch_size,
                        "learning_rate": config.learning_rate,
                        "patience": config.patience,
                        "uses_lengths": fit.name == "sequence_lstm",
                    },
                    metrics=merged[fit.name],
                )
            )
        lstm = models["sequence_lstm"]
        assert isinstance(lstm, SequenceLSTM)
        merged["sequence_lstm"]["padding_invariance_max_logit_delta"] = (
            self._padding_invariance_delta(lstm, padded[test_idx], lengths[test_idx])
        )
        merged["sequence_lstm"]["accuracy_gain_over_padded_mlp"] = (
            merged["sequence_lstm"]["test_accuracy"] - merged["padded_mlp"]["test_accuracy"]
        )

        candidates = tuple(
            CandidateResult(name=name, metrics=metrics) for name, metrics in merged.items()
        )
        records_json, records_csv = _write_candidate_records(context, tuple(records))
        artifacts["candidate_records_json"] = records_json
        artifacts["candidate_records_csv"] = records_csv

        plot_context = (
            f"seed {context.seed} | {config.n_sequences} sequences | "
            f"lengths {config.min_length}-{config.max_length} | Adam lr "
            f"{config.learning_rate} | patience {config.patience}"
        )
        artifacts["training_loss_plot"] = training_curves_plot(
            curves,
            ylabel="cross-entropy",
            title="Temporal-XOR training and validation loss",
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/sequence_training_loss.png",
        )
        artifacts["validation_accuracy_plot"] = training_curves_plot(
            accuracy_curves,
            ylabel="validation accuracy",
            title="Temporal-XOR validation accuracy",
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/sequence_validation_accuracy.png",
        )
        artifacts["accuracy_by_length_plot"] = accuracy_by_length_plot(
            bucket_labels,
            bucket_accuracies,
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/sequence_accuracy_by_length.png",
        )

        source = SourceMetadata(
            kind=SourceKind.GENERATOR,
            name="temporal_xor",
            version="1.0.0",
            fingerprint_sha256=array_fingerprint(
                dataset.sequences, dataset.lengths, dataset.targets
            ),
            target_used_for_fit=True,
            details={
                "n_sequences": config.n_sequences,
                "min_length": config.min_length,
                "max_length": config.max_length,
                "token_alphabet": [-1.0, 1.0],
                "label_rule": "first token XOR last token",
            },
        )
        selected_result = next(candidate for candidate in candidates if candidate.name == selected)
        return RunResult(
            experiment=config.experiment,
            paradigm=LearningParadigm.DEEP,
            seed=context.seed,
            source=source,
            selected_model=selected,
            metrics=dict(selected_result.metrics),
            candidates=candidates,
            artifacts=artifacts,
            notes=(
                "The padded MLP receives the same optimizer, budget, and data; its "
                "failure isolates what recurrence adds on variable-length structure.",
                "Packed sequences keep padding out of the LSTM cell; appended padding "
                "leaves its logits unchanged up to the recorded delta.",
            ),
        )


# ------------------------------------------------------------------------------- #29


class DeepAutoencoderBenchmark:
    """Bottleneck-autoencoder anomaly scoring against a linear PCA baseline."""

    def __init__(self, config: DeepAutoencoderBenchmarkConfig) -> None:
        self._config = config

    @staticmethod
    def _corrupt_partition(
        flattened: FloatArray,
        partition: IntArray,
        *,
        fraction: float,
        seed: int,
    ) -> FloatArray:
        rng = generator_for_seed(seed)
        count = max(1, round(fraction * len(partition)))
        source = rng.choice(partition, size=count, replace=False)
        return np.asarray(
            np.stack([rng.permutation(flattened[index]) for index in source]),
            dtype=np.float64,
        )

    @_preserve_torch_state
    def run(self, context: RunContext) -> RunResult:
        config = self._config
        images, targets = _load_digit_images(
            config.n_samples,
            seed=derive_named_seed(context.seed, config.experiment, "data"),
            sampling="uniform",
        )
        flattened = images.reshape(len(images), -1)
        train_idx, validation_idx, test_idx = _three_way_indices(
            np.arange(len(flattened), dtype=np.int64),
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=derive_named_seed(context.seed, config.experiment, "split"),
            stratify=False,
        )
        inputs = torch.from_numpy(flattened.astype(np.float32))

        model_seed = derive_named_seed(
            context.seed, config.experiment, "candidate", "bottleneck_autoencoder"
        )
        seed_torch(model_seed)
        model = BottleneckAutoencoder(
            n_features=flattened.shape[1],
            hidden_units=config.hidden_units,
            latent_dim=config.latent_dim,
        )
        trainer = Trainer(
            TrainerConfig(
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=derive_named_seed(
                    context.seed, config.experiment, "trainer", "bottleneck_autoencoder"
                ),
                patience=config.patience,
                min_delta=1e-6,
                device=config.device,
            )
        )
        objective = ReconstructionObjective()
        report = trainer.fit(
            model,
            tensor_dataset(inputs[train_idx]),
            tensor_dataset(inputs[validation_idx]),
            objective,
        )

        validation_corrupted = self._corrupt_partition(
            flattened,
            validation_idx,
            fraction=config.anomaly_fraction,
            seed=derive_named_seed(context.seed, config.experiment, "validation-anomalies"),
        )
        n_validation_anomalies = len(validation_corrupted)
        validation_corrupted_inputs = torch.from_numpy(validation_corrupted.astype(np.float32))

        validation_inlier_errors = reconstruction_errors(model, inputs[validation_idx]).numpy()
        validation_anomaly_errors = reconstruction_errors(
            model, validation_corrupted_inputs
        ).numpy()
        validation_score_labels = np.concatenate(
            [
                np.zeros(len(validation_inlier_errors)),
                np.ones(len(validation_anomaly_errors)),
            ]
        )
        validation_scores = np.concatenate(
            [validation_inlier_errors, validation_anomaly_errors]
        ).astype(np.float64)
        validation_auroc = roc_auc_score(validation_score_labels, validation_scores)

        pca = PCA(n_components=config.latent_dim)
        pca.fit(flattened[train_idx])
        pca_validation_inlier = self._pca_errors(pca, flattened[validation_idx])
        pca_validation_anomaly = self._pca_errors(pca, validation_corrupted)
        pca_validation_scores = np.concatenate([pca_validation_inlier, pca_validation_anomaly])
        pca_validation_auroc = roc_auc_score(validation_score_labels, pca_validation_scores)

        selection_candidates = (
            CandidateResult(
                name="bottleneck_autoencoder",
                metrics={"selection_validation_auroc": validation_auroc},
            ),
            CandidateResult(
                name="pca_reconstruction",
                metrics={"selection_validation_auroc": pca_validation_auroc},
            ),
        )
        selected = _select_candidate(
            selection_candidates,
            metric="selection_validation_auroc",
            prefer="bottleneck_autoencoder",
        )

        # Independent test anomalies and retrospective class labels are opened now.
        corrupted = self._corrupt_partition(
            flattened,
            test_idx,
            fraction=config.anomaly_fraction,
            seed=derive_named_seed(context.seed, config.experiment, "test-anomalies"),
        )
        n_anomalies = len(corrupted)
        corrupted_inputs = torch.from_numpy(corrupted.astype(np.float32))
        inlier_errors = reconstruction_errors(model, inputs[test_idx]).numpy()
        anomaly_errors = reconstruction_errors(model, corrupted_inputs).numpy()
        score_labels = np.concatenate([np.zeros(len(inlier_errors)), np.ones(len(anomaly_errors))])
        scores = np.concatenate([inlier_errors, anomaly_errors]).astype(np.float64)
        auroc = roc_auc_score(score_labels, scores)
        pca_inlier = self._pca_errors(pca, flattened[test_idx])
        pca_anomaly = self._pca_errors(pca, corrupted)
        pca_scores = np.concatenate([pca_inlier, pca_anomaly])
        pca_auroc = roc_auc_score(score_labels, pca_scores)

        latent = _encode_latent(model, inputs[test_idx]).numpy().astype(np.float64)
        if config.latent_dim > 2:
            latent_2d = PCA(n_components=2).fit_transform(latent)
            latent_axis = "latent PC"
        else:
            latent_2d = latent
            latent_axis = "latent dim"
        silhouette = silhouette_analysis(latent, targets[test_idx])

        merged: dict[str, dict[str, float]] = {
            "bottleneck_autoencoder": {
                "selection_validation_auroc": validation_auroc,
                "anomaly_auroc": auroc,
                "mean_inlier_error": float(np.mean(inlier_errors)),
                "mean_anomaly_error": float(np.mean(anomaly_errors)),
                "error_separation_ratio": float(
                    np.mean(anomaly_errors) / max(np.mean(inlier_errors), 1e-12)
                ),
                "latent_class_silhouette": silhouette.score,
                "best_epoch": float(report.best_epoch),
                "epochs_run": float(len(report.history)),
                "stopped_early": float(report.stopped_early),
                "parameter_count": float(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
            },
            "pca_reconstruction": {
                "selection_validation_auroc": pca_validation_auroc,
                "anomaly_auroc": pca_auroc,
                "mean_inlier_error": float(np.mean(pca_inlier)),
                "mean_anomaly_error": float(np.mean(pca_anomaly)),
                "error_separation_ratio": float(
                    float(np.mean(pca_anomaly)) / max(float(np.mean(pca_inlier)), 1e-12)
                ),
            },
        }
        candidates = tuple(
            CandidateResult(name=name, metrics=metrics) for name, metrics in merged.items()
        )

        records = tuple(
            _CandidateRecord(
                study="digits_anomaly",
                candidate=name,
                seed=model_seed if name == "bottleneck_autoencoder" else None,
                parameters={
                    "latent_dim": config.latent_dim,
                    "hidden_units": config.hidden_units
                    if name == "bottleneck_autoencoder"
                    else None,
                    "corruption": "per-image pixel permutation",
                    "anomaly_fraction": config.anomaly_fraction,
                },
                metrics=metrics,
            )
            for name, metrics in merged.items()
        )
        records_json, records_csv = _write_candidate_records(context, records)

        artifacts: dict[str, str] = {
            "candidate_records_json": records_json,
            "candidate_records_csv": records_csv,
            "autoencoder_state": _save_state_dict(
                context, model, "models/bottleneck_autoencoder_state.pt"
            ),
        }
        plot_context = (
            f"seed {context.seed} | {config.n_samples} digits | latent {config.latent_dim} | "
            f"{n_anomalies} pixel-permuted anomalies | Adam lr {config.learning_rate}"
        )
        artifacts["training_loss_plot"] = training_curves_plot(
            {"bottleneck_autoencoder": _loss_curves(report)},
            ylabel="mean-squared reconstruction error",
            title="Autoencoder optimization",
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/autoencoder_training_loss.png",
            log_scale=True,
        )
        show = min(6, len(test_idx), n_anomalies)
        reconstructed = _predict_logits(model, inputs[test_idx][:show]).numpy().reshape(show, 8, 8)
        corrupted_reconstructed = (
            _predict_logits(model, corrupted_inputs[:show]).numpy().reshape(show, 8, 8)
        )
        artifacts["reconstruction_plot"] = image_panel_plot(
            {
                "inlier": images[test_idx][:show],
                "inlier_reconstruction": reconstructed.astype(np.float64),
                "anomaly": corrupted[:show].reshape(show, 8, 8),
                "anomaly_reconstruction": corrupted_reconstructed.astype(np.float64),
            },
            {
                "inlier": [f"digit {targets[test_idx][i]}" for i in range(show)],
                "inlier_reconstruction": [f"mse {inlier_errors[i]:.4f}" for i in range(show)],
                "anomaly": ["permuted"] * show,
                "anomaly_reconstruction": [f"mse {anomaly_errors[i]:.4f}" for i in range(show)],
            },
            title="The bottleneck reconstructs structure and rejects shuffled pixels",
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/autoencoder_reconstructions.png",
        )
        artifacts["error_histogram_plot"] = error_histogram_plot(
            inlier_errors.astype(np.float64),
            anomaly_errors.astype(np.float64),
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/autoencoder_error_histogram.png",
        )
        ae_curve = binary_roc_curve(score_labels, scores)
        pca_curve = binary_roc_curve(score_labels, pca_scores)
        artifacts["roc_curve_plot"] = roc_curves_plot(
            {
                "bottleneck_autoencoder": (ae_curve[0], ae_curve[1], auroc),
                "pca_reconstruction": (pca_curve[0], pca_curve[1], pca_auroc),
            },
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/autoencoder_roc.png",
        )
        artifacts["latent_scatter_plot"] = latent_scatter_plot(
            latent_2d,
            np.asarray(targets[test_idx], dtype=np.int64),
            axis_label=latent_axis,
            context=plot_context,
            artifacts=context.artifacts,
            relative_path="plots/autoencoder_latent.png",
        )

        source = SourceMetadata(
            kind=SourceKind.DATASET,
            name="sklearn_digits_8x8",
            version=version("scikit-learn"),
            fingerprint_sha256=array_fingerprint(images, targets),
            target_used_for_fit=False,
            details={
                "n_samples_used": len(targets),
                "n_train": len(train_idx),
                "n_validation": len(validation_idx),
                "n_test_inliers": len(test_idx),
                "n_injected_anomalies": int(n_anomalies),
                "n_validation_anomalies": int(n_validation_anomalies),
                "labels_reserved_for_retrospective_evaluation": True,
            },
        )
        selected_result = next(candidate for candidate in candidates if candidate.name == selected)
        return RunResult(
            experiment=config.experiment,
            paradigm=LearningParadigm.DEEP,
            seed=context.seed,
            source=source,
            selected_model=selected,
            metrics=dict(selected_result.metrics),
            candidates=candidates,
            artifacts=artifacts,
            notes=(
                "Training, model selection, and anomaly scoring never see class labels; "
                "labels color the latent view and score silhouette retrospectively.",
                "Synthetic anomalies from the validation partition select the method; "
                "the independently corrupted test partition is opened once afterward.",
                "The PCA baseline shares the latent budget, isolating what the "
                "nonlinearity contributes to anomaly separation.",
            ),
        )

    @staticmethod
    def _pca_errors(pca: PCA, batch: FloatArray) -> FloatArray:
        reconstructed = pca.inverse_transform(pca.transform(batch))
        return np.asarray(np.mean((batch - reconstructed) ** 2, axis=1), dtype=np.float64)
