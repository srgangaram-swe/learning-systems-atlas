"""Architecture tests keep unsupervised mathematics framework-independent."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

FROM_SCRATCH_MODULES = (
    "base.py",
    "datasets.py",
    "decomposition.py",
    "density.py",
    "evaluation.py",
    "hierarchical.py",
    "kmeans.py",
    "manifold.py",
    "metrics.py",
    "mixture.py",
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


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_from_scratch_modules_have_no_framework_or_experiment_dependencies() -> None:
    root = Path("src/learning_atlas/unsupervised")
    for filename in FROM_SCRATCH_MODULES:
        path = root / filename
        assert path.is_file(), f"missing from-scratch module: {path}"
        forbidden = {
            dependency
            for dependency in _imported_modules(path)
            if any(
                dependency == root_name or dependency.startswith(f"{root_name}.")
                for root_name in FORBIDDEN_ROOTS
            )
        }
        assert not forbidden, f"{path} crosses the from-scratch boundary: {sorted(forbidden)}"
