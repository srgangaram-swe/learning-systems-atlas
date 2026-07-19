"""Architecture tests keep from-scratch mathematics auditable and reusable."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

FROM_SCRATCH_MODULES = (
    "baselines.py",
    "datasets.py",
    "ensemble.py",
    "linear_model.py",
    "metrics.py",
    "model_selection.py",
    "naive_bayes.py",
    "neighbors.py",
    "preprocessing.py",
    "svm.py",
    "tree.py",
)
FORBIDDEN_ROOTS = {
    "joblib",
    "matplotlib",
    "sklearn",
    "learning_atlas.cli",
    "learning_atlas.core.artifacts",
    "learning_atlas.core.config",
    "learning_atlas.reporting",
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_from_scratch_modules_have_no_framework_or_experiment_dependencies() -> None:
    root = Path("src/learning_atlas/supervised")
    for filename in FROM_SCRATCH_MODULES:
        path = root / filename
        assert path.is_file(), f"missing from-scratch module: {path}"
        forbidden = {
            dependency
            for dependency in imported_modules(path)
            if any(
                dependency == root_name or dependency.startswith(f"{root_name}.")
                for root_name in FORBIDDEN_ROOTS
            )
        }
        assert not forbidden, f"{path} crosses the from-scratch boundary: {sorted(forbidden)}"
