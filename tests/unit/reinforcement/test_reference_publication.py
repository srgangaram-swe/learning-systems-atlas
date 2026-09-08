"""The public evidence export is an explicit, hash-checked allowlist."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from learning_atlas.core.artifacts import ArtifactStore

pytestmark = pytest.mark.unit
SCRIPT = Path("scripts/publish_reinforcement_reference.py")


def publisher():
    spec = importlib.util.spec_from_file_location("reference_export", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    store = ArtifactStore(root)
    plots = (
        "bandit_posteriors",
        "bandit_rewards",
        "bernoulli_regret",
        "cliff_learning",
        "cliff_risk",
        "dqn_ablation",
        "dqn_td_loss",
        "gaussian_regret",
        "gridworld_policy",
        "heldout_return_distribution",
        "heldout_seed_means",
        "interaction_cost",
        "navigation_learning",
        "neural_learning",
        "neural_validation",
        "planning_residuals",
        "ppo_optimization",
        "reinforce_gradient_variance",
        "sample_efficiency",
        "tabular_policy_gap",
    )
    for name in plots:
        store.write_text(f"plots/{name}.png", "synthetic-byte-fixture")
    for name in ("evidence.json", "comparison.csv", "comparison.md", "resolved_config.json"):
        store.write_text(name, "{}")
    store.write_json("result.json", {"experiment": "reinforcement_benchmark"})
    store.write_json(
        "manifest.json",
        {
            "experiment": "reinforcement_benchmark",
            "environment": {"git": {"dirty": False, "commit": "a" * 40}},
            "artifacts_sha256": {str(path): store.sha256(str(path)) for path in store.files()},
        },
    )
    return root


def test_export_is_exact_and_never_copies_checkpoint(source, tmp_path):
    (source / "secret-checkpoint.json").write_text("must-not-publish")
    destination = tmp_path / "published"
    publisher().publish(source, destination)
    assert len(list(destination.rglob("*.png"))) == 20
    assert not (destination / "secret-checkpoint.json").exists()
    assert (destination / "evidence.json").read_bytes() == (source / "evidence.json").read_bytes()
    with pytest.raises(FileExistsError):
        publisher().publish(source, destination)


@pytest.mark.parametrize(
    "fault", ["wrong_experiment", "dirty", "extra", "checksum", "symlink", "oversized"]
)
def test_export_rejects_unverifiable_evidence(source, tmp_path, fault):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if fault == "wrong_experiment":
        manifest["experiment"] = "other"
    elif fault == "dirty":
        manifest["environment"]["git"]["dirty"] = True
    elif fault == "extra":
        manifest["artifacts_sha256"]["checkpoint.json"] = "0" * 64
    elif fault == "checksum":
        manifest["artifacts_sha256"]["evidence.json"] = "0" * 64
    elif fault == "symlink":
        (source / "evidence.json").unlink()
        (source / "evidence.json").symlink_to(tmp_path / "outside")
    else:
        with (source / "evidence.json").open("wb") as stream:
            stream.truncate(17 * 1024 * 1024)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        publisher().publish(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_export_copy_failure_cleans_staging(source, tmp_path, monkeypatch):
    module = publisher()

    def fail(*args):
        raise OSError("injected copy failure")

    monkeypatch.setattr(module.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="injected"):
        module.publish(source, tmp_path / "failed")
    assert not list(tmp_path.glob(".failed.*"))


def test_export_cli_error_is_bounded(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "missing"), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "reference publication failed" in result.stderr
    assert "Traceback" not in result.stderr
