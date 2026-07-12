"""Run a deterministic collection of validated reference experiments."""

import os
import shutil
import tempfile
from pathlib import Path

from learning_atlas.core.config import ExperimentConfig, load_config
from learning_atlas.core.contracts import RunResult
from learning_atlas.core.runner import run_experiment


def discover_configs(config_dir: Path) -> tuple[Path, ...]:
    """Discover YAML configs recursively in stable path order."""

    if not config_dir.is_dir():
        msg = f"configuration directory does not exist: {config_dir}"
        raise NotADirectoryError(msg)
    configs = tuple(sorted((*config_dir.rglob("*.yaml"), *config_dir.rglob("*.yml"))))
    if not configs:
        msg = f"no YAML configurations found under: {config_dir}"
        raise FileNotFoundError(msg)
    return configs


def run_suite(config_dir: Path, output_dir: Path) -> tuple[RunResult, ...]:
    """Preflight every config, then atomically publish the complete suite."""

    target = output_dir.resolve()
    if target.exists() and any(target.iterdir()):
        msg = f"refusing to overwrite non-empty suite directory: {target}"
        raise FileExistsError(msg)

    configs: list[ExperimentConfig] = []
    seen_names: set[str] = set()
    for config_path in discover_configs(config_dir):
        config = load_config(config_path)
        if config.experiment in seen_names:
            msg = f"suite contains duplicate experiment name: {config.experiment}"
            raise ValueError(msg)
        seen_names.add(config.experiment)
        configs.append(config)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    results: list[RunResult] = []
    try:
        for config in configs:
            results.append(run_experiment(config, staging / config.experiment))
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return tuple(results)
