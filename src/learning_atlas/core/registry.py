"""Explicit experiment registry; no dynamic import strings or hidden plugin magic."""

from collections.abc import Callable
from dataclasses import dataclass

from learning_atlas.core.config import (
    BaseExperimentConfig,
    ClassificationBenchmarkConfig,
    ClusteringBenchmarkConfig,
    ExperimentConfig,
    QLearningConfig,
    RegressionBenchmarkConfig,
    ScratchClassificationBenchmarkConfig,
    ScratchRegressionBenchmarkConfig,
)
from learning_atlas.core.contracts import Experiment, LearningParadigm


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Discoverable metadata and a type-checked experiment factory."""

    name: str
    paradigm: LearningParadigm
    description: str
    factory: Callable[[BaseExperimentConfig], Experiment]


def _regression(config: BaseExperimentConfig) -> Experiment:
    if not isinstance(config, RegressionBenchmarkConfig):
        raise TypeError("regression_benchmark requires RegressionBenchmarkConfig")
    from learning_atlas.supervised.regression import RegressionBenchmark

    return RegressionBenchmark(config)


def _clustering(config: BaseExperimentConfig) -> Experiment:
    if not isinstance(config, ClusteringBenchmarkConfig):
        raise TypeError("clustering_benchmark requires ClusteringBenchmarkConfig")
    from learning_atlas.unsupervised.clustering import ClusteringBenchmark

    return ClusteringBenchmark(config)


def _classification(config: BaseExperimentConfig) -> Experiment:
    if not isinstance(config, ClassificationBenchmarkConfig):
        raise TypeError("classification_benchmark requires ClassificationBenchmarkConfig")
    from learning_atlas.supervised.classification import ClassificationBenchmark

    return ClassificationBenchmark(config)


def _q_learning(config: BaseExperimentConfig) -> Experiment:
    if not isinstance(config, QLearningConfig):
        raise TypeError("q_learning_frozen_lake requires QLearningConfig")
    from learning_atlas.reinforcement.q_learning import FrozenLakeBenchmark

    return FrozenLakeBenchmark(config)


def _scratch_regression(config: BaseExperimentConfig) -> Experiment:
    if not isinstance(config, ScratchRegressionBenchmarkConfig):
        raise TypeError("scratch_regression_benchmark requires ScratchRegressionBenchmarkConfig")
    from learning_atlas.supervised.comparison import ScratchRegressionBenchmark

    return ScratchRegressionBenchmark(config)


def _scratch_classification(config: BaseExperimentConfig) -> Experiment:
    if not isinstance(config, ScratchClassificationBenchmarkConfig):
        raise TypeError(
            "scratch_classification_benchmark requires ScratchClassificationBenchmarkConfig"
        )
    from learning_atlas.supervised.comparison import ScratchClassificationBenchmark

    return ScratchClassificationBenchmark(config)


REGISTRY: dict[str, ExperimentSpec] = {
    "regression_benchmark": ExperimentSpec(
        name="regression_benchmark",
        paradigm=LearningParadigm.SUPERVISED,
        description="Leakage-safe dummy, Ridge, and random-forest regression benchmark.",
        factory=_regression,
    ),
    "classification_benchmark": ExperimentSpec(
        name="classification_benchmark",
        paradigm=LearningParadigm.SUPERVISED,
        description="Leakage-safe dummy, logistic, and random-forest classification benchmark.",
        factory=_classification,
    ),
    "scratch_regression_benchmark": ExperimentSpec(
        name="scratch_regression_benchmark",
        paradigm=LearningParadigm.SUPERVISED,
        description="From-scratch linear, neighbor, tree, forest, and boosting regression.",
        factory=_scratch_regression,
    ),
    "scratch_classification_benchmark": ExperimentSpec(
        name="scratch_classification_benchmark",
        paradigm=LearningParadigm.SUPERVISED,
        description="From-scratch linear, kernel, Bayesian, neighbor, tree, and ensemble models.",
        factory=_scratch_classification,
    ),
    "clustering_benchmark": ExperimentSpec(
        name="clustering_benchmark",
        paradigm=LearningParadigm.UNSUPERVISED,
        description="Label-isolated K-means and DBSCAN structure-discovery benchmark.",
        factory=_clustering,
    ),
    "q_learning_frozen_lake": ExperimentSpec(
        name="q_learning_frozen_lake",
        paradigm=LearningParadigm.REINFORCEMENT,
        description="Random-policy baseline and tabular Q-learning on FrozenLake.",
        factory=_q_learning,
    ),
}


def build_experiment(config: ExperimentConfig) -> Experiment:
    """Construct the experiment selected by a validated discriminated config."""

    try:
        spec = REGISTRY[config.experiment]
    except KeyError as error:  # defensive: validation normally makes this unreachable
        msg = f"unregistered experiment: {config.experiment}"
        raise ValueError(msg) from error
    return spec.factory(config)
