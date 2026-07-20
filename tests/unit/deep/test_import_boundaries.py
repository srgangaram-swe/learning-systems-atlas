"""Architecture tests keep from-scratch deep-learning mathematics isolated."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

FROM_SCRATCH_MODULES = ("autograd.py", "datasets.py", "mlp.py")
FORBIDDEN_ROOTS = {
    "joblib",
    "matplotlib",
    "sklearn",
    "torch",
    "learning_atlas.cli",
    "learning_atlas.core.artifacts",
    "learning_atlas.core.config",
    "learning_atlas.deep.autoencoder",
    "learning_atlas.deep.benchmarks",
    "learning_atlas.deep.sequence",
    "learning_atlas.deep.training",
    "learning_atlas.deep.vision",
    "learning_atlas.reporting",
}
AUTOGRAD_ALLOWED_ROOTS = {"__future__", "collections", "typing", "numpy"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_from_scratch_modules_do_not_cross_framework_or_experiment_boundaries() -> None:
    root = Path("src/learning_atlas/deep")
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
        assert not forbidden, f"{path} crosses the boundary: {sorted(forbidden)}"


def test_autograd_depends_only_on_numpy_and_the_standard_library() -> None:
    path = Path("src/learning_atlas/deep/autograd.py")
    unexpected = {
        dependency
        for dependency in _imported_modules(path)
        if dependency.split(".", maxsplit=1)[0] not in AUTOGRAD_ALLOWED_ROOTS
    }

    assert not unexpected, f"autograd gained external dependencies: {sorted(unexpected)}"


def test_from_scratch_modules_never_seed_or_sample_from_numpy_global_rng() -> None:
    root = Path("src/learning_atlas/deep")
    forbidden_calls = {
        "np.random.seed",
        "np.random.rand",
        "np.random.randn",
        "np.random.random",
        "np.random.choice",
        "np.random.permutation",
    }
    violations: dict[str, list[str]] = {}
    for filename in FROM_SCRATCH_MODULES:
        path = root / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            parts: list[str] = []
            target = node.func
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                parts.append(target.id)
            qualified = ".".join(reversed(parts))
            if qualified in forbidden_calls:
                calls.append(qualified)
        if calls:
            violations[filename] = sorted(calls)

    assert not violations, f"global RNG use found: {violations}"
