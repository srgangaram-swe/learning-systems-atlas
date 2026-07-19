"""Strict configuration parsing tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from learning_atlas.core.config import (
    ClassificationBenchmarkConfig,
    ClusteringBenchmarkConfig,
    QLearningConfig,
    RegressionBenchmarkConfig,
    config_schema,
    load_config,
    parse_config,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("path", "expected_type"),
    [
        ("configs/supervised/regression.yaml", RegressionBenchmarkConfig),
        ("configs/supervised/classification.yaml", ClassificationBenchmarkConfig),
        ("configs/unsupervised/clustering.yaml", ClusteringBenchmarkConfig),
        ("configs/reinforcement/q_learning.yaml", QLearningConfig),
    ],
)
def test_committed_configs_are_valid(path: str, expected_type: type[object]) -> None:
    assert isinstance(load_config(Path(path)), expected_type)


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        parse_config({"experiment": "regression_benchmark", "sead": 42})


def test_unknown_discriminator_is_rejected() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        parse_config({"experiment": "imaginary_model"})


@pytest.mark.parametrize(
    "raw",
    [
        {"experiment": "regression_benchmark", "seed": "42"},
        {"experiment": "q_learning_frozen_lake", "is_slippery": "false"},
    ],
)
def test_config_does_not_coerce_strings(raw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_config(raw)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config)


def test_numeric_boundaries_are_validated() -> None:
    with pytest.raises(ValidationError, match="greater_than_equal"):
        QLearningConfig(training_episodes=99)
    with pytest.raises(ValidationError, match="less_than_equal"):
        ClusteringBenchmarkConfig(noise=0.6)
    with pytest.raises(ValidationError, match="less_than"):
        RegressionBenchmarkConfig(test_size=0.5)
    with pytest.raises(ValidationError, match="epsilon_end"):
        QLearningConfig(epsilon_start=0.1, epsilon_end=0.2)


def test_schema_exposes_all_registered_discriminators() -> None:
    serialized = str(config_schema())
    assert "regression_benchmark" in serialized
    assert "classification_benchmark" in serialized
    assert "clustering_benchmark" in serialized
    assert "q_learning_frozen_lake" in serialized
