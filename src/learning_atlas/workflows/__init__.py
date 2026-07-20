"""Cross-paradigm orchestration workflows."""

from learning_atlas.workflows.deep_benchmark import run_deep_benchmark
from learning_atlas.workflows.suite import discover_configs, run_suite
from learning_atlas.workflows.supervised_benchmark import run_supervised_benchmark
from learning_atlas.workflows.unsupervised_benchmark import run_unsupervised_benchmark

__all__ = [
    "discover_configs",
    "run_deep_benchmark",
    "run_suite",
    "run_supervised_benchmark",
    "run_unsupervised_benchmark",
]
