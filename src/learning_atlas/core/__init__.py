"""Shared contracts and execution infrastructure."""

from learning_atlas.core.config import ExperimentConfig, load_config
from learning_atlas.core.contracts import LearningParadigm, RunResult
from learning_atlas.core.estimators import (
    ClassifierMixin,
    Estimator,
    NotFittedError,
    RegressorMixin,
)
from learning_atlas.core.runner import run_experiment

__all__ = [
    "ClassifierMixin",
    "Estimator",
    "ExperimentConfig",
    "LearningParadigm",
    "NotFittedError",
    "RegressorMixin",
    "RunResult",
    "load_config",
    "run_experiment",
]
