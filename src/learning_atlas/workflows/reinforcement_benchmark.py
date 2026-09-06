"""Sprint 5 entry point reusing the existing atomic experiment transaction."""

from pathlib import Path

from learning_atlas.core.config import ReinforcementBenchmarkConfig, load_config
from learning_atlas.core.contracts import RunResult
from learning_atlas.core.runner import run_experiment


def run_reinforcement_benchmark(config_path: Path, output_dir: Path) -> RunResult:
    """Validate the exact sprint experiment before any model or output allocation."""

    config = load_config(config_path)
    if not isinstance(config, ReinforcementBenchmarkConfig):
        raise ValueError("reinforcement benchmark requires ReinforcementBenchmarkConfig")
    return run_experiment(config, output_dir)
