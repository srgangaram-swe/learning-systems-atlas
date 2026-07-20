"""Scientific-boundary and helper tests for Sprint 4 benchmark experiments."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import (
    DeepAutoencoderBenchmarkConfig,
    DeepSequenceBenchmarkConfig,
    DeepVisionBenchmarkConfig,
    ScratchMLPBenchmarkConfig,
    load_config,
)
from learning_atlas.core.contracts import CandidateResult, RunContext
from learning_atlas.core.registry import build_experiment
from learning_atlas.core.reproducibility import derive_named_seed
from learning_atlas.deep.autoencoder import BottleneckAutoencoder
from learning_atlas.deep.benchmarks import (
    DeepAutoencoderBenchmark,
    DeepSequenceBenchmark,
    DeepVisionBenchmark,
    ScratchMLPBenchmark,
    _encode_latent,
    _load_digit_images,
    _predict_logits,
    _preserve_torch_state,
    _save_state_dict,
    _select_candidate,
    _three_way_indices,
)
from learning_atlas.deep.sequence import SequenceLSTM
from learning_atlas.deep.vision import CompactCNN

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("filename", "expected_type", "benchmark_type"),
    [
        ("scratch-mlp.yaml", ScratchMLPBenchmarkConfig, ScratchMLPBenchmark),
        ("vision.yaml", DeepVisionBenchmarkConfig, DeepVisionBenchmark),
        ("sequence.yaml", DeepSequenceBenchmarkConfig, DeepSequenceBenchmark),
        ("autoencoder.yaml", DeepAutoencoderBenchmarkConfig, DeepAutoencoderBenchmark),
    ],
)
def test_committed_configs_dispatch_through_public_registry(
    filename: str, expected_type: type[object], benchmark_type: type[object]
) -> None:
    config = load_config(Path("configs/deep/sprint-04") / filename)

    assert isinstance(config, expected_type)
    assert isinstance(build_experiment(config), benchmark_type)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_fraction": 0.4, "test_size": 0.4},
        {"min_length": 10, "max_length": 9},
        {"hidden_units": 64, "latent_dim": 8},
        {"hidden_units": 8, "latent_dim": 8},
        {"validation_fraction": 0.4, "test_fraction": 0.4},
    ],
)
def test_deep_config_cross_field_contracts(kwargs: dict[str, object]) -> None:
    constructor: (
        type[ScratchMLPBenchmarkConfig]
        | type[DeepSequenceBenchmarkConfig]
        | type[DeepAutoencoderBenchmarkConfig]
        | type[DeepVisionBenchmarkConfig]
    )
    if "test_size" in kwargs:
        constructor = ScratchMLPBenchmarkConfig
    elif "min_length" in kwargs:
        constructor = DeepSequenceBenchmarkConfig
    elif "latent_dim" in kwargs:
        constructor = DeepAutoencoderBenchmarkConfig
    else:
        constructor = DeepVisionBenchmarkConfig
    with pytest.raises(ValueError):
        constructor(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("count", [200, 201, 997, 1_796, 1_797])
def test_digit_subsample_has_exact_size_all_classes_and_replays(count: int) -> None:
    first_images, first_targets = _load_digit_images(count, seed=18)
    second_images, second_targets = _load_digit_images(count, seed=18)

    assert first_images.shape == (count, 8, 8)
    assert first_targets.shape == (count,)
    assert np.array_equal(np.unique(first_targets), np.arange(10))
    assert np.array_equal(first_images, second_images)
    assert np.array_equal(first_targets, second_targets)
    assert not first_images.flags.writeable
    assert not first_targets.flags.writeable


def test_digit_loader_rejects_invalid_counts() -> None:
    for invalid in (0, True, 1.5, 1_798):
        with pytest.raises(ValueError, match=r"n_samples|requested"):
            _load_digit_images(invalid, seed=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sampling"):
        _load_digit_images(200, seed=2, sampling="target_ranked")  # type: ignore[arg-type]


def test_uniform_digit_sampling_is_exact_replayable_and_label_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sklearn.datasets

    images = np.arange(40 * 8 * 8, dtype=np.float64).reshape(40, 8, 8)
    labels = np.arange(40, dtype=np.int64) % 10
    monkeypatch.setattr(
        sklearn.datasets,
        "load_digits",
        lambda: SimpleNamespace(images=images, target=labels),
    )
    first_images, first_targets = _load_digit_images(17, seed=91, sampling="uniform")
    monkeypatch.setattr(
        sklearn.datasets,
        "load_digits",
        lambda: SimpleNamespace(images=images, target=(labels + 1) % 10),
    )
    second_images, second_targets = _load_digit_images(17, seed=91, sampling="uniform")

    assert first_images.shape == (17, 8, 8)
    assert np.array_equal(first_images, second_images)
    assert np.array_equal((first_targets + 1) % 10, second_targets)
    assert not first_images.flags.writeable
    assert not first_targets.flags.writeable


def test_three_way_split_is_disjoint_complete_stratified_and_repeatable() -> None:
    targets = np.repeat(np.arange(3, dtype=np.int64), 20)
    first = _three_way_indices(
        targets,
        validation_fraction=0.2,
        test_fraction=0.25,
        seed=88,
        stratify=True,
    )
    second = _three_way_indices(
        targets,
        validation_fraction=0.2,
        test_fraction=0.25,
        seed=88,
        stratify=True,
    )

    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]).isdisjoint(first[2])
    assert set(first[1]).isdisjoint(first[2])
    assert np.array_equal(np.sort(np.concatenate(first)), np.arange(len(targets)))
    for partition in first:
        assert np.array_equal(np.unique(targets[partition]), np.arange(3))


def test_unstratified_three_way_split_depends_only_on_row_count() -> None:
    first = _three_way_indices(
        np.arange(60, dtype=np.int64),
        validation_fraction=0.2,
        test_fraction=0.25,
        seed=88,
        stratify=False,
    )
    second = _three_way_indices(
        np.full(60, 999, dtype=np.int64),
        validation_fraction=0.2,
        test_fraction=0.25,
        seed=88,
        stratify=False,
    )

    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))


def test_digit_benchmarks_declare_label_use_at_sampling_and_split_boundaries() -> None:
    vision_source = inspect.getsource(DeepVisionBenchmark.run)
    autoencoder_source = inspect.getsource(DeepAutoencoderBenchmark.run)

    assert 'sampling="label_stratified"' in vision_source
    assert 'sampling="uniform"' in autoencoder_source
    assert "np.arange(len(flattened), dtype=np.int64)" in autoencoder_source


@pytest.mark.parametrize(
    ("targets", "validation_fraction", "test_fraction", "message"),
    [
        (np.asarray([[0, 1], [1, 0]]), 0.2, 0.2, "one-dimensional"),
        (np.asarray([0, 1]), 0.2, 0.2, "at least 3"),
        (np.arange(10), 0.0, 0.2, "validation_fraction"),
        (np.arange(10), 0.2, np.nan, "test_fraction"),
        (np.arange(10), 0.5, 0.5, "below 1"),
        (np.asarray([0, 0, 0]), 0.1, 0.1, "empty partition"),
    ],
)
def test_three_way_split_rejects_invalid_partitions(
    targets: np.ndarray, validation_fraction: float, test_fraction: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _three_way_indices(
            targets,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=1,
            stratify=False,
        )


def test_selection_uses_declared_metric_and_preferred_tie() -> None:
    candidates = (
        CandidateResult(name="baseline", metrics={"validation": 0.8}),
        CandidateResult(name="model", metrics={"validation": 0.9}),
    )
    assert _select_candidate(candidates, metric="validation", prefer="baseline") == "model"

    tied = (
        CandidateResult(name="baseline", metrics={"validation": 0.9}),
        CandidateResult(name="model", metrics={"validation": 0.9}),
    )
    assert _select_candidate(tied, metric="validation", prefer="model") == "model"


def test_selection_rejects_empty_candidates_or_missing_preference() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        _select_candidate((), metric="validation", prefer="model")

    candidates = (CandidateResult(name="model", metrics={"validation": 0.9}),)
    with pytest.raises(ValueError, match="preferred candidate 'baseline' is not present"):
        _select_candidate(candidates, metric="validation", prefer="baseline")


@pytest.mark.parametrize(
    "candidate",
    [
        CandidateResult(name="model", metrics={"validation": 0.9, "test": 1.0}),
        CandidateResult.model_construct(name="model", metrics={}),
    ],
)
def test_selection_rejects_metrics_outside_exact_declared_boundary(
    candidate: CandidateResult,
) -> None:
    with pytest.raises(ValueError, match="must expose only selection metric 'validation'"):
        _select_candidate((candidate,), metric="validation", prefer="model")


def test_selection_rejects_non_finite_values_even_for_unvalidated_records() -> None:
    candidate = CandidateResult.model_construct(name="model", metrics={"validation": np.nan})

    with pytest.raises(ValueError, match="non-finite selection metric"):
        _select_candidate((candidate,), metric="validation", prefer="model")


@pytest.mark.parametrize(
    ("benchmark", "expected_metric", "post_selection_markers"),
    [
        (
            ScratchMLPBenchmark,
            "mean_validation_accuracy",
            ("fit.test_indices", "self._gradient_check(", "self._probability_panel("),
        ),
        (
            DeepVisionBenchmark,
            "selection_validation_accuracy",
            ("test_dataset = tensor_dataset(", "targets[test_idx]", "_checkpoint_reload_deltas("),
        ),
        (
            DeepSequenceBenchmark,
            "selection_validation_accuracy",
            (
                "test_partition = tensor_dataset(",
                "dataset.lengths[test_idx]",
                "_padding_invariance_delta(",
            ),
        ),
        (
            DeepAutoencoderBenchmark,
            "selection_validation_auroc",
            ('"test-anomalies"', "inputs[test_idx]", "silhouette_analysis("),
        ),
    ],
)
def test_benchmarks_freeze_validation_only_selection_before_held_out_work(
    benchmark: type[object], expected_metric: str, post_selection_markers: tuple[str, ...]
) -> None:
    source = textwrap.dedent(inspect.getsource(benchmark.run))  # type: ignore[attr-defined]
    if benchmark is not ScratchMLPBenchmark:
        assert source.startswith("@_preserve_torch_state")
    tree = ast.parse(source)
    selected_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "selected" for target in node.targets)
    )
    assert isinstance(selected_assignment.value, ast.Call)
    metric_keyword = next(
        keyword for keyword in selected_assignment.value.keywords if keyword.arg == "metric"
    )
    assert isinstance(metric_keyword.value, ast.Constant)
    assert metric_keyword.value.value == expected_metric

    selection_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "selection_candidates"
            for target in node.targets
        )
    )
    candidate_calls = [
        node
        for node in ast.walk(selection_assignment.value)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CandidateResult"
    ]
    assert candidate_calls
    for call in candidate_calls:
        metrics_keyword = next(keyword for keyword in call.keywords if keyword.arg == "metrics")
        assert isinstance(metrics_keyword.value, ast.Dict)
        assert [
            key.value for key in metrics_keyword.value.keys if isinstance(key, ast.Constant)
        ] == [expected_metric]

    selection_offset = source.index("selected = _select_candidate(")
    assert 'metric="test_' not in source
    for marker in post_selection_markers:
        assert source.index(marker) > selection_offset


def test_torch_state_guard_restores_rng_and_determinism_on_success_and_failure() -> None:
    original_rng = torch.get_rng_state().clone()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    @_preserve_torch_state
    def mutate_state(*, fail: bool) -> None:
        torch.manual_seed(9001)
        torch.rand(4)
        torch.use_deterministic_algorithms(False)
        if fail:
            raise RuntimeError("injected failure")

    try:
        torch.manual_seed(77)
        torch.use_deterministic_algorithms(True, warn_only=True)
        expected_rng = torch.get_rng_state().clone()

        mutate_state(fail=False)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0.0, atol=0.0)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

        with pytest.raises(RuntimeError, match="injected failure"):
            mutate_state(fail=True)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0.0, atol=0.0)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.set_rng_state(original_rng)
        torch.use_deterministic_algorithms(
            original_deterministic,
            warn_only=original_warn_only,
        )


def test_scratch_candidate_records_publish_actual_task_model_seeds(tmp_path: Path) -> None:
    config = ScratchMLPBenchmarkConfig(
        n_samples=64,
        hidden_units=2,
        batch_size=64,
        max_epochs=1,
        gradient_check_samples=4,
    )
    context = RunContext(seed=19, artifacts=ArtifactStore(tmp_path))

    result = ScratchMLPBenchmark(config).run(context)

    records = json.loads((tmp_path / result.artifacts["candidate_records_json"]).read_text())
    csv_records = (tmp_path / result.artifacts["candidate_records_csv"]).read_text()
    assert len(records) == 2
    for record in records:
        candidate = record["candidate"]
        assert record["seed"] is None
        expected = {
            task: derive_named_seed(19, config.experiment, "candidate", candidate, task)
            for task in ("xor", "two_moons")
        }
        assert record["named_seeds"] == expected
        assert all(f"{task}={seed}" in csv_records for task, seed in expected.items())


def test_prediction_and_encoding_helpers_restore_mode_and_return_cpu() -> None:
    cnn = CompactCNN(channels=2, hidden_units=4)
    cnn.train()
    logits = _predict_logits(cnn, torch.zeros(3, 1, 8, 8))
    assert logits.shape == (3, 10)
    assert logits.device.type == "cpu"
    assert cnn.training

    autoencoder = BottleneckAutoencoder(hidden_units=16, latent_dim=4)
    autoencoder.eval()
    latent = _encode_latent(autoencoder, torch.zeros(3, 64))
    assert latent.shape == (3, 4)
    assert latent.device.type == "cpu"
    assert not autoencoder.training


def test_state_dict_publication_is_byte_stable(tmp_path: Path) -> None:
    model = SequenceLSTM(hidden_size=4)
    first = RunContext(seed=1, artifacts=ArtifactStore(tmp_path / "first"))
    second = RunContext(seed=1, artifacts=ArtifactStore(tmp_path / "second"))

    relative = "models/state.pt"
    _save_state_dict(first, model, relative)
    _save_state_dict(second, model, relative)

    assert first.artifacts.sha256(relative) == second.artifacts.sha256(relative)


def test_autoencoder_corruption_and_pca_errors_are_deterministic() -> None:
    flattened = np.arange(12 * 4, dtype=np.float64).reshape(12, 4)
    partition = np.arange(4, 12, dtype=np.int64)
    first = DeepAutoencoderBenchmark._corrupt_partition(flattened, partition, fraction=0.5, seed=3)
    second = DeepAutoencoderBenchmark._corrupt_partition(flattened, partition, fraction=0.5, seed=3)
    assert first.shape == (4, 4)
    assert np.array_equal(first, second)
    assert all(sorted(row) in [sorted(source) for source in flattened[partition]] for row in first)


def test_autoencoder_validation_and_test_corruptions_use_isolated_named_streams() -> None:
    flattened = np.arange(24 * 8, dtype=np.float64).reshape(24, 8)
    validation = np.arange(0, 8, dtype=np.int64)
    test = np.arange(8, 16, dtype=np.int64)
    validation_seed = derive_named_seed(42, "deep_autoencoder_benchmark", "validation-anomalies")
    test_seed = derive_named_seed(42, "deep_autoencoder_benchmark", "test-anomalies")

    test_before_validation = DeepAutoencoderBenchmark._corrupt_partition(
        flattened, test, fraction=0.75, seed=test_seed
    )
    validation_corrupted = DeepAutoencoderBenchmark._corrupt_partition(
        flattened, validation, fraction=0.75, seed=validation_seed
    )
    test_after_validation = DeepAutoencoderBenchmark._corrupt_partition(
        flattened, test, fraction=0.75, seed=test_seed
    )

    assert validation_seed != test_seed
    assert np.array_equal(test_before_validation, test_after_validation)
    validation_row_sums = set(flattened[validation].sum(axis=1))
    test_row_sums = set(flattened[test].sum(axis=1))
    assert set(validation_corrupted.sum(axis=1)) <= validation_row_sums
    assert set(test_after_validation.sum(axis=1)) <= test_row_sums
    assert validation_row_sums.isdisjoint(test_row_sums)
