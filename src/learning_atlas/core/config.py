"""Strict, discriminated configuration models for reference experiments."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class StrictConfig(BaseModel):
    """Base model that rejects misspelled or unsupported configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BaseExperimentConfig(StrictConfig):
    """Fields shared by every experiment."""

    experiment: str
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class RegressionBenchmarkConfig(BaseExperimentConfig):
    """Configuration for the supervised regression benchmark."""

    experiment: Literal["regression_benchmark"] = "regression_benchmark"
    dataset: Literal["diabetes"] = "diabetes"
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    ridge_alpha: float = Field(default=10.0, gt=0.0)
    forest_estimators: int = Field(default=160, ge=10, le=2_000)
    forest_max_depth: int | None = Field(default=8, ge=1, le=100)


class ClassificationBenchmarkConfig(BaseExperimentConfig):
    """Configuration for the supervised classification benchmark."""

    experiment: Literal["classification_benchmark"] = "classification_benchmark"
    dataset: Literal["breast_cancer"] = "breast_cancer"
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    logistic_c: float = Field(default=1.0, gt=0.0)
    forest_estimators: int = Field(default=160, ge=10, le=2_000)
    forest_max_depth: int | None = Field(default=8, ge=1, le=100)


class ScratchRegressionBenchmarkConfig(BaseExperimentConfig):
    """Configuration for the from-scratch regression comparison."""

    experiment: Literal["scratch_regression_benchmark"] = "scratch_regression_benchmark"
    n_samples: int = Field(default=480, ge=120, le=20_000)
    n_features: int = Field(default=10, ge=2, le=100)
    n_informative: int = Field(default=7, ge=1, le=100)
    noise: float = Field(default=12.0, ge=0.0, le=1_000.0)
    test_size: float = Field(default=0.25, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    forest_estimators: int = Field(default=48, ge=5, le=500)
    boosting_estimators: int = Field(default=60, ge=5, le=500)
    max_depth: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def informative_features_cannot_exceed_total(self) -> Self:
        """Reject an impossible synthetic regression specification."""

        if self.n_informative > self.n_features:
            msg = "n_informative must be less than or equal to n_features"
            raise ValueError(msg)
        return self


class ScratchClassificationBenchmarkConfig(BaseExperimentConfig):
    """Configuration for the from-scratch classification comparison."""

    experiment: Literal["scratch_classification_benchmark"] = "scratch_classification_benchmark"
    n_samples: int = Field(default=520, ge=160, le=20_000)
    n_features: int = Field(default=8, ge=2, le=100)
    n_informative: int = Field(default=5, ge=2, le=100)
    class_sep: float = Field(default=1.8, gt=0.0, le=20.0)
    label_noise: float = Field(default=0.03, ge=0.0, lt=0.5)
    test_size: float = Field(default=0.25, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    forest_estimators: int = Field(default=48, ge=5, le=500)
    boosting_estimators: int = Field(default=60, ge=5, le=500)
    max_depth: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def informative_features_cannot_exceed_total(self) -> Self:
        """Reject an impossible synthetic classification specification."""

        if self.n_informative > self.n_features:
            msg = "n_informative must be less than or equal to n_features"
            raise ValueError(msg)
        return self


class ClusteringBenchmarkConfig(BaseExperimentConfig):
    """Configuration for the unsupervised structure-discovery benchmark."""

    experiment: Literal["clustering_benchmark"] = "clustering_benchmark"
    dataset: Literal["noisy_moons"] = "noisy_moons"
    n_samples: int = Field(default=600, ge=100, le=100_000)
    noise: float = Field(default=0.08, ge=0.0, le=0.5)
    kmeans_clusters: int = Field(default=2, ge=2, le=20)
    kmeans_n_init: int = Field(default=20, ge=1, le=100)
    dbscan_eps: float = Field(default=0.25, gt=0.0)
    dbscan_min_samples: int = Field(default=8, ge=2, le=1_000)


class QLearningConfig(BaseExperimentConfig):
    """Configuration for tabular Q-learning on FrozenLake."""

    experiment: Literal["q_learning_frozen_lake"] = "q_learning_frozen_lake"
    environment: Literal["FrozenLake-v1"] = "FrozenLake-v1"
    map_name: Literal["4x4", "8x8"] = "4x4"
    is_slippery: bool = True
    training_episodes: int = Field(default=12_000, ge=100, le=1_000_000)
    evaluation_episodes: int = Field(default=1_000, ge=50, le=100_000)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    discount_factor: float = Field(default=0.99, ge=0.0, le=1.0)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.9995, gt=0.0, le=1.0)
    max_steps_per_episode: int = Field(default=100, ge=1, le=10_000)

    @model_validator(mode="after")
    def epsilon_floor_cannot_exceed_start(self) -> Self:
        """Reject schedules that would increase exploration unexpectedly."""

        if self.epsilon_end > self.epsilon_start:
            msg = "epsilon_end must be less than or equal to epsilon_start"
            raise ValueError(msg)
        return self


ExperimentConfig = Annotated[
    RegressionBenchmarkConfig
    | ClassificationBenchmarkConfig
    | ScratchRegressionBenchmarkConfig
    | ScratchClassificationBenchmarkConfig
    | ClusteringBenchmarkConfig
    | QLearningConfig,
    Field(discriminator="experiment"),
]

_CONFIG_ADAPTER: TypeAdapter[ExperimentConfig] = TypeAdapter(ExperimentConfig)


def parse_config(raw: object) -> ExperimentConfig:
    """Validate an already-decoded experiment configuration."""

    return _CONFIG_ADAPTER.validate_python(raw)


def load_config(path: Path) -> ExperimentConfig:
    """Load and strictly validate a YAML experiment configuration."""

    try:
        with path.open(encoding="utf-8") as stream:
            raw: object = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        msg = f"invalid YAML configuration: {path}"
        raise ValueError(msg) from error
    if not isinstance(raw, dict):
        msg = f"configuration must be a YAML mapping: {path}"
        raise ValueError(msg)
    return parse_config(raw)


def config_schema() -> dict[str, object]:
    """Return the JSON schema for the discriminated configuration union."""

    return _CONFIG_ADAPTER.json_schema()
