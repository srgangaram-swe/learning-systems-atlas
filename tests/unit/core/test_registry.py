"""Explicit experiment-registry tests."""

import pytest

from learning_atlas.core.config import (
    ClassificationBenchmarkConfig,
    ClusteringBenchmarkConfig,
    DeepAutoencoderBenchmarkConfig,
    DeepSequenceBenchmarkConfig,
    DeepVisionBenchmarkConfig,
    QLearningConfig,
    RegressionBenchmarkConfig,
    ReinforcementBenchmarkConfig,
    ScratchClassificationBenchmarkConfig,
    ScratchClusteringBenchmarkConfig,
    ScratchMLPBenchmarkConfig,
    ScratchRegressionBenchmarkConfig,
    ScratchRepresentationBenchmarkConfig,
)
from learning_atlas.core.contracts import Experiment
from learning_atlas.core.registry import REGISTRY, build_experiment

pytestmark = pytest.mark.unit


def test_registry_covers_every_committed_experiment() -> None:
    configs = (
        RegressionBenchmarkConfig(),
        ClassificationBenchmarkConfig(),
        ScratchRegressionBenchmarkConfig(),
        ScratchClassificationBenchmarkConfig(),
        ClusteringBenchmarkConfig(),
        ScratchClusteringBenchmarkConfig(),
        ScratchRepresentationBenchmarkConfig(),
        ScratchMLPBenchmarkConfig(),
        DeepVisionBenchmarkConfig(),
        DeepSequenceBenchmarkConfig(),
        DeepAutoencoderBenchmarkConfig(),
        QLearningConfig(),
        ReinforcementBenchmarkConfig(),
    )
    assert {config.experiment for config in configs} == set(REGISTRY)
    assert all(isinstance(build_experiment(config), Experiment) for config in configs)


def test_registry_metadata_is_discoverable() -> None:
    assert {spec.paradigm.value for spec in REGISTRY.values()} == {
        "supervised",
        "unsupervised",
        "deep",
        "reinforcement",
    }
    assert all(spec.description.endswith(".") for spec in REGISTRY.values())
