"""Reproducibility controls and environment provenance."""

import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class GitState:
    """Source revision recorded with every run."""

    commit: str | None
    dirty: bool | None


def _validate_seed(seed: int) -> None:
    if not 0 <= seed <= 2**32 - 1:
        msg = "seed must fit in an unsigned 32-bit integer"
        raise ValueError(msg)


def generator_for_seed(seed: int) -> np.random.Generator:
    """Create an explicit NumPy generator without mutating process-global RNG state."""

    _validate_seed(seed)
    return np.random.default_rng(seed)


def derive_seed(seed: int, stream: int) -> int:
    """Derive a deterministic uint32 seed for an independent named random stream."""

    _validate_seed(seed)
    if stream < 0:
        msg = "stream must be non-negative"
        raise ValueError(msg)
    sequence = np.random.SeedSequence([seed, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def git_state(cwd: Path | None = None) -> GitState:
    """Read the current Git revision without making execution depend on Git."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return GitState(commit=None, dirty=None)
    return GitState(commit=commit, dirty=bool(status.strip()))


def environment_metadata() -> dict[str, JsonValue]:
    """Collect runtime facts needed to interpret or reproduce numerical results."""

    packages: dict[str, JsonValue] = {}
    for package in (
        "gymnasium",
        "joblib",
        "matplotlib",
        "numpy",
        "pydantic",
        "PyYAML",
        "scikit-learn",
        "typer",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"

    source = GitState(commit=None, dirty=None)
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            source = git_state(parent)
            break
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "git": {"commit": source.commit, "dirty": source.dirty},
    }
