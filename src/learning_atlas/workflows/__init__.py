"""Cross-paradigm orchestration workflows."""

from learning_atlas.workflows.suite import discover_configs, run_suite
from learning_atlas.workflows.supervised_benchmark import run_supervised_benchmark

__all__ = ["discover_configs", "run_suite", "run_supervised_benchmark"]
