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


class ScratchClusteringBenchmarkConfig(BaseExperimentConfig):
    """Configuration for the from-scratch clustering and stability study."""

    experiment: Literal["scratch_clustering_benchmark"] = "scratch_clustering_benchmark"
    n_samples: int = Field(default=240, ge=120, le=2_000)
    stability_trials: int = Field(default=4, ge=2, le=20)
    perturbation_scale: float = Field(default=0.015, gt=0.0, le=0.25)
    kmeans_n_init: int = Field(default=8, ge=2, le=50)
    kmeans_max_iter: int = Field(default=200, ge=10, le=2_000)
    gmm_n_init: int = Field(default=3, ge=1, le=20)
    gmm_max_iter: int = Field(default=150, ge=10, le=2_000)
    dbscan_eps: float = Field(default=0.24, gt=0.0, le=5.0)
    dbscan_min_samples: int = Field(default=6, ge=2, le=100)


class ScratchRepresentationBenchmarkConfig(BaseExperimentConfig):
    """Configuration for PCA and t-SNE representation evidence."""

    experiment: Literal["scratch_representation_benchmark"] = "scratch_representation_benchmark"
    n_samples: int = Field(default=180, ge=90, le=1_000)
    n_features: int = Field(default=6, ge=3, le=50)
    clusters: int = Field(default=4, ge=2, le=12)
    pca_components: int = Field(default=2, ge=2, le=20)
    tsne_perplexity: float = Field(default=22.0, gt=2.0, le=100.0)
    tsne_iterations: int = Field(default=400, ge=250, le=2_000)

    @model_validator(mode="after")
    def representation_dimensions_must_be_feasible(self) -> Self:
        """Reject representations that exceed their feature/sample rank."""

        if self.pca_components > self.n_features:
            msg = "pca_components must be less than or equal to n_features"
            raise ValueError(msg)
        if self.tsne_perplexity >= self.n_samples:
            msg = "tsne_perplexity must be less than n_samples"
            raise ValueError(msg)
        if self.clusters > self.n_samples:
            msg = "clusters must be less than or equal to n_samples"
            raise ValueError(msg)
        return self


class ScratchMLPBenchmarkConfig(BaseExperimentConfig):
    """Configuration for the autograd MLP study on planar tasks."""

    experiment: Literal["scratch_mlp_benchmark"] = "scratch_mlp_benchmark"
    n_samples: int = Field(default=320, ge=64, le=20_000)
    xor_noise: float = Field(default=0.15, ge=0.0, le=1.0)
    moons_noise: float = Field(default=0.1, ge=0.0, le=1.0)
    hidden_units: int = Field(default=16, ge=2, le=256)
    activation: Literal["relu", "tanh"] = "tanh"
    learning_rate: float = Field(default=0.3, gt=0.0, le=10.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    batch_size: int = Field(default=32, ge=1, le=4_096)
    max_epochs: int = Field(default=300, ge=1, le=10_000)
    validation_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    test_size: float = Field(default=0.25, gt=0.05, lt=0.5)
    gradient_check_samples: int = Field(default=16, ge=4, le=128)

    @model_validator(mode="after")
    def holdouts_must_leave_training_data(self) -> Self:
        """Reserve distinct validation and untouched-test partitions."""

        if self.validation_fraction + self.test_size >= 0.8:
            msg = "validation_fraction plus test_size must be below 0.8"
            raise ValueError(msg)
        return self


class _TorchStudyConfig(BaseExperimentConfig):
    """Shared knobs for every Trainer-driven PyTorch study."""

    validation_fraction: float = Field(default=0.16, gt=0.05, lt=0.5)
    test_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    max_epochs: int = Field(default=30, ge=1, le=1_000)
    batch_size: int = Field(default=64, ge=1, le=4_096)
    learning_rate: float = Field(default=3e-3, gt=0.0, le=1.0)
    patience: int = Field(default=6, ge=1, le=100)
    device: Literal["cpu", "auto"] = "cpu"

    @model_validator(mode="after")
    def holdouts_must_leave_training_data(self) -> Self:
        """Reject splits that starve the training partition."""

        if self.validation_fraction + self.test_fraction >= 0.8:
            msg = "validation_fraction plus test_fraction must be below 0.8"
            raise ValueError(msg)
        return self


class DeepVisionBenchmarkConfig(_TorchStudyConfig):
    """Configuration for the CNN-versus-MLP digits study."""

    experiment: Literal["deep_vision_benchmark"] = "deep_vision_benchmark"
    n_samples: int = Field(default=1_000, ge=200, le=1_797)
    hidden_units: int = Field(default=32, ge=4, le=512)
    channels: int = Field(default=8, ge=2, le=64)
    max_epochs: int = Field(default=25, ge=1, le=1_000)


class DeepSequenceBenchmarkConfig(_TorchStudyConfig):
    """Configuration for the LSTM long-range dependency study."""

    experiment: Literal["deep_sequence_benchmark"] = "deep_sequence_benchmark"
    n_sequences: int = Field(default=1_200, ge=200, le=50_000)
    min_length: int = Field(default=6, ge=2, le=512)
    max_length: int = Field(default=24, ge=2, le=512)
    hidden_size: int = Field(default=48, ge=4, le=512)
    mlp_hidden_units: int = Field(default=64, ge=4, le=1_024)
    max_epochs: int = Field(default=60, ge=1, le=1_000)
    learning_rate: float = Field(default=5e-3, gt=0.0, le=1.0)
    patience: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def lengths_must_form_a_range(self) -> Self:
        """Reject inverted or degenerate sequence-length ranges."""

        if self.max_length < self.min_length:
            msg = "max_length must be greater than or equal to min_length"
            raise ValueError(msg)
        return self


class DeepAutoencoderBenchmarkConfig(_TorchStudyConfig):
    """Configuration for the bottleneck autoencoder anomaly study."""

    experiment: Literal["deep_autoencoder_benchmark"] = "deep_autoencoder_benchmark"
    n_samples: int = Field(default=1_200, ge=200, le=1_797)
    hidden_units: int = Field(default=32, ge=4, le=512)
    latent_dim: int = Field(default=8, ge=2, le=64)
    anomaly_fraction: float = Field(default=0.5, gt=0.05, le=1.0)
    max_epochs: int = Field(default=60, ge=1, le=1_000)
    patience: int = Field(default=8, ge=1, le=100)

    @model_validator(mode="after")
    def bottleneck_must_compress(self) -> Self:
        """Reject architectures that are not genuine bottlenecks."""

        if self.latent_dim >= self.hidden_units:
            msg = "latent_dim must be strictly smaller than hidden_units"
            raise ValueError(msg)
        if self.hidden_units >= 64:
            msg = "hidden_units must be smaller than the 64-feature digits input"
            raise ValueError(msg)
        return self


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


class ReinforcementBenchmarkConfig(BaseExperimentConfig):
    """Bounded multi-seed Sprint 5 laboratory, with a permanently separate test stream."""

    experiment: Literal["reinforcement_benchmark"] = "reinforcement_benchmark"
    repetitions: int = Field(default=3, ge=1, le=20)
    bandit_steps: int = Field(default=2000, ge=10, le=100_000)
    tabular_episodes: int = Field(default=1500, ge=10, le=10_000)
    reinforce_episodes: int = Field(default=2000, ge=8, le=4000)
    dqn_episodes: int = Field(default=800, ge=8, le=4000)
    ppo_episodes: int = Field(default=300, ge=8, le=4000)
    control_max_steps: int = Field(default=500, ge=10, le=500)
    validation_episodes: int = Field(default=10, ge=1, le=100)
    test_episodes: int = Field(default=100, ge=2, le=500)
    variance_episodes: int = Field(default=32, ge=2, le=128)
    dqn_ablations: bool = True

    @model_validator(mode="after")
    def bound_total_interactions(self) -> Self:
        """Bound the combined run, not merely each individual candidate."""

        dqn_factor = 4 if self.dqn_ablations else 1
        neural = (
            self.repetitions
            * self.control_max_steps
            * (self.reinforce_episodes + dqn_factor * self.dqn_episodes + self.ppo_episodes)
        )
        tabular = self.repetitions * self.tabular_episodes * 4 * 500
        if neural > 20_000_000 or tabular > 30_000_000:
            raise ValueError("combined reinforcement interaction budget is too large")
        return self


ExperimentConfig = Annotated[
    RegressionBenchmarkConfig
    | ClassificationBenchmarkConfig
    | ScratchRegressionBenchmarkConfig
    | ScratchClassificationBenchmarkConfig
    | ClusteringBenchmarkConfig
    | ScratchClusteringBenchmarkConfig
    | ScratchRepresentationBenchmarkConfig
    | ScratchMLPBenchmarkConfig
    | DeepVisionBenchmarkConfig
    | DeepSequenceBenchmarkConfig
    | DeepAutoencoderBenchmarkConfig
    | QLearningConfig
    | ReinforcementBenchmarkConfig,
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
