"""Explicit experiment-registry tests."""

import pytest

from learning_atlas.core.config import (
    ClassificationBenchmarkConfig,
    ClusteringBenchmarkConfig,
    QLearningConfig,
    RegressionBenchmarkConfig,
)
from learning_atlas.core.contracts import Experiment
from learning_atlas.core.registry import REGISTRY, build_experiment

pytestmark = pytest.mark.unit


def test_registry_covers_every_committed_experiment() -> None:
    configs = (
        RegressionBenchmarkConfig(),
        ClassificationBenchmarkConfig(),
        ClusteringBenchmarkConfig(),
        QLearningConfig(),
    )
    assert {config.experiment for config in configs} == set(REGISTRY)
    assert all(isinstance(build_experiment(config), Experiment) for config in configs)


def test_registry_metadata_is_discoverable() -> None:
    assert {spec.paradigm.value for spec in REGISTRY.values()} == {
        "supervised",
        "unsupervised",
        "reinforcement",
    }
    assert all(spec.description.endswith(".") for spec in REGISTRY.values())
