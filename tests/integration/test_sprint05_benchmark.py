"""Real CPU learning, transaction, CLI, seed isolation, and test-selection contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from typer.testing import CliRunner

from learning_atlas.cli import app
from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import (
    RegressionBenchmarkConfig,
    ReinforcementBenchmarkConfig,
    load_config,
)
from learning_atlas.core.contracts import RunContext
from learning_atlas.core.registry import _reinforcement
from learning_atlas.core.runner import run_experiment
from learning_atlas.reinforcement.benchmark import ReinforcementBenchmark, control_configs
from learning_atlas.reinforcement.control import evaluate_control
from learning_atlas.reinforcement.diagnostics import baseline_variance
from learning_atlas.reinforcement.dqn import train_dqn
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.policy_gradient import train_policy_gradient
from learning_atlas.reinforcement.validation import ReinforcementError
from learning_atlas.workflows.reinforcement_benchmark import run_reinforcement_benchmark

pytestmark = pytest.mark.integration
CI_CONFIG = Path("tests/fixtures/reinforcement_ci.yaml")


def test_full_reduced_benchmark_is_deterministic_and_transactional(tmp_path):
    first = run_reinforcement_benchmark(CI_CONFIG, tmp_path / "first")
    second = run_reinforcement_benchmark(CI_CONFIG, tmp_path / "second")
    assert first == second
    assert len([path for path in first.artifacts.values() if path.endswith(".png")]) == 20
    manifest_a = json.loads((tmp_path / "first/manifest.json").read_text())
    manifest_b = json.loads((tmp_path / "second/manifest.json").read_text())
    assert manifest_a["artifacts_sha256"] == manifest_b["artifacts_sha256"]
    assert len(manifest_a["artifacts_sha256"]) == 25
    evidence = json.loads((tmp_path / "first/evidence.json").read_text())
    assert set(evidence["tables"]) >= {
        "bandits",
        "planning",
        "tabular",
        "gradient_variance",
        "test_returns",
    }
    with pytest.raises(FileExistsError):
        run_reinforcement_benchmark(CI_CONFIG, tmp_path / "first")
    assert not list(tmp_path.glob(".first.*"))


@pytest.mark.parametrize(
    "method,episodes,rate,minimum",
    [
        ("reinforce", 400, 0.005, 65.0),
        ("dqn", 300, 0.001, 100.0),
        ("ppo", 80, 0.001, 60.0),
    ],
)
def test_reduced_real_control_learns_over_independent_random_policy(
    method, episodes, rate, minimum
):
    config = ControlConfig(
        method=method, episodes=episodes, learning_rate=rate, validation_episodes=5
    )
    report = train_dqn(config) if method == "dqn" else train_policy_gradient(config)
    # This integration fixture has its own evaluation seed, distinct from the
    # committed reference holdout. Thresholds express broad learning, not goldens.
    learned = evaluate_control(report.model, seed=771, episodes=20).mean()
    random = evaluate_control(None, seed=771, episodes=20).mean()
    assert learned > minimum
    assert learned > 2 * random
    assert 0 < report.selected_episode <= episodes


@pytest.mark.parametrize("method", ["reinforce", "ppo"])
def test_policy_gradient_repeatability_global_isolation_and_baseline_ablation(method):
    config = ControlConfig(
        method=method,
        episodes=16,
        max_steps=40,
        validation_episodes=2,
        learned_baseline=False,
        target_kl=1e-8,
    )
    before = torch.get_rng_state().clone()
    first = train_policy_gradient(config)
    train_policy_gradient(config.model_copy(update={"seed": 9}))
    second = train_policy_gradient(config)
    assert first.returns == second.returns
    assert first.diagnostics == second.diagnostics
    assert torch.equal(before, torch.get_rng_state())
    for key, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, second.model.state_dict()[key])
    if method == "ppo":
        assert any(record["kl_early_stop"] for record in first.diagnostics)
    variance = baseline_variance(first.model, config, episodes=3)
    assert variance["without_baseline"] >= 0
    assert variance["independent_trajectories"] == 3


def test_registry_config_and_workflow_reject_wrong_contracts(tmp_path):
    with pytest.raises(TypeError):
        _reinforcement(RegressionBenchmarkConfig())
    with pytest.raises(ValueError, match="requires"):
        run_reinforcement_benchmark(Path("configs/supervised/regression.yaml"), tmp_path / "wrong")
    with pytest.raises(ValueError, match="seed"):
        ReinforcementBenchmark(ReinforcementBenchmarkConfig()).run(
            RunContext(9, ArtifactStore(tmp_path))
        )
    with pytest.raises(ValidationError, match="budget"):
        ReinforcementBenchmarkConfig(repetitions=20, reinforce_episodes=4000)
    no_ablation = ReinforcementBenchmarkConfig(repetitions=1, dqn_ablations=False)
    assert len(control_configs(no_ablation)) == 3


def test_neural_failure_removes_all_staged_classical_and_partial_evidence(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ReinforcementError("injected numerical failure")

    monkeypatch.setattr("learning_atlas.reinforcement.benchmark.train_policy_gradient", fail)
    output = tmp_path / "failed"
    with pytest.raises(ReinforcementError, match="injected"):
        run_reinforcement_benchmark(CI_CONFIG, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.*"))
    cli = CliRunner().invoke(app, ["benchmark-reinforcement", str(CI_CONFIG), "-o", str(output)])
    assert cli.exit_code == 2
    assert "injected numerical failure" in cli.stderr


def test_cli_dispatch_runs_real_experiment_and_wrong_profile_fails(tmp_path):
    result = CliRunner().invoke(
        app, ["benchmark-reinforcement", str(CI_CONFIG), "-o", str(tmp_path / "cli")]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["experiment"] == "reinforcement_benchmark"
    wrong = CliRunner().invoke(
        app,
        [
            "benchmark-reinforcement",
            "configs/supervised/regression.yaml",
            "-o",
            str(tmp_path / "wrong"),
        ],
    )
    assert wrong.exit_code == 2


def test_changing_test_returns_cannot_change_selected_policy(tmp_path, monkeypatch):
    import learning_atlas.reinforcement.benchmark as benchmark

    config = load_config(CI_CONFIG).model_copy(update={"dqn_ablations": False})
    monkeypatch.setattr(benchmark, "publish_reinforcement_plots", lambda *args: {})
    first = run_experiment(config, tmp_path / "first")
    monkeypatch.setattr(
        benchmark, "evaluate_control", lambda *args, **kwargs: np.full(kwargs["episodes"], 1.0)
    )
    second = run_experiment(config, tmp_path / "second")
    assert first.selected_model == second.selected_model
    assert first.metrics["validation_return"] == second.metrics["validation_return"]
    assert second.metrics["test_return"] == 1.0
