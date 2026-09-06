"""One transactional experiment connecting bandit, planning, TD, and neural evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from learning_atlas.core.config import ReinforcementBenchmarkConfig
from learning_atlas.core.contracts import (
    CandidateResult,
    LearningParadigm,
    RunContext,
    RunResult,
    SourceKind,
    SourceMetadata,
)
from learning_atlas.core.reproducibility import derive_seed
from learning_atlas.reinforcement.bandits import BanditMethod, Distribution, simulate_bandit
from learning_atlas.reinforcement.control import ControlReport, evaluate_control
from learning_atlas.reinforcement.diagnostics import baseline_variance
from learning_atlas.reinforcement.dqn import train_dqn
from learning_atlas.reinforcement.gridworld import Gridworld
from learning_atlas.reinforcement.neural_config import ControlConfig
from learning_atlas.reinforcement.planning import evaluate_policy, plan
from learning_atlas.reinforcement.policy_gradient import train_policy_gradient
from learning_atlas.reinforcement.tabular import train_tabular
from learning_atlas.reporting.reinforcement import publish_reinforcement_plots

Row = dict[str, str | float]


@dataclass(slots=True)
class LaboratoryEvidence:
    """Redistribution-safe synthetic aggregates; no model weights or executable state."""

    tables: dict[str, list[Row]] = field(default_factory=dict)

    def add(self, table: str, **row: str | float) -> None:
        self.tables.setdefault(table, []).append(row)


def _window_rows(values: np.ndarray, *, width: int = 25) -> list[tuple[int, float]]:
    """Non-overlapping windows include the final partial window, with no smoothing look-ahead."""

    return [
        (min(start + width, len(values)), float(values[start : start + width].mean()))
        for start in range(0, len(values), width)
    ]


def _classical(config: ReinforcementBenchmarkConfig, evidence: LaboratoryEvidence) -> None:
    distributions: tuple[Distribution, ...] = ("bernoulli", "gaussian")
    methods: tuple[BanditMethod, ...] = ("epsilon_greedy", "ucb", "thompson")
    for repetition in range(config.repetitions):
        seed = (config.seed + repetition) % 2**32
        for distribution in distributions:
            for method in methods:
                trace = simulate_bandit(
                    np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                    distribution=distribution,
                    method=method,
                    steps=config.bandit_steps,
                    seed=seed,
                )
                stride = max(1, config.bandit_steps // 250)
                reward_mean = np.cumsum(trace.rewards) / np.arange(1, len(trace.rewards) + 1)
                indices = sorted({*range(0, config.bandit_steps, stride), config.bandit_steps - 1})
                for index in indices:
                    evidence.add(
                        "bandits",
                        distribution=distribution,
                        method=method,
                        seed=float(seed),
                        step=float(index + 1),
                        regret=float(trace.cumulative_regret[index]),
                        mean_reward=float(reward_mean[index]),
                    )
                if method == "thompson":
                    for arm, (first, second) in enumerate(
                        zip(trace.posterior_first, trace.posterior_second, strict=True)
                    ):
                        mean = first / (first + second) if distribution == "bernoulli" else first
                        variance = (
                            first * second / ((first + second) ** 2 * (first + second + 1))
                            if distribution == "bernoulli"
                            else second
                        )
                        evidence.add(
                            "posteriors",
                            distribution=distribution,
                            seed=float(seed),
                            arm=float(arm),
                            mean=float(mean),
                            posterior_sd=float(np.sqrt(variance)),
                            truth=float(0.1 + 0.2 * arm),
                        )
        for task, layout in (
            ("navigation", ("S...", ".##.", "...G")),
            ("cliff", ("............", "............", "............", "SCCCCCCCCCCG")),
        ):
            environment = Gridworld(layout)
            mdp = environment.model()
            optimal = plan(mdp, gamma=0.95)
            tabular_methods: tuple[Literal["q_learning", "sarsa"], ...] = ("q_learning", "sarsa")
            for td_method in tabular_methods:
                td_trace = train_tabular(
                    environment,
                    method=td_method,
                    episodes=config.tabular_episodes,
                    seed=seed,
                    epsilon_start=0.1 if task == "cliff" else 0.3,
                    epsilon_end=0.1 if task == "cliff" else 0.05,
                    epsilon_decay=1.0 if task == "cliff" else 0.995,
                )
                for (episode, reward), (_, falls) in zip(
                    _window_rows(td_trace.returns), _window_rows(td_trace.cliff_falls), strict=True
                ):
                    evidence.add(
                        "tabular",
                        task=task,
                        method=td_method,
                        seed=float(seed),
                        episode=float(episode),
                        mean_return=reward,
                        mean_cliff_falls=falls,
                    )
                policy = td_trace.q_values.argmax(axis=1)
                values = evaluate_policy(mdp, policy, 0.95)
                evidence.add(
                    "policy_gap",
                    task=task,
                    method=td_method,
                    seed=float(seed),
                    start_value_gap=float(
                        optimal.values[environment.start] - values[environment.start]
                    ),
                )
    grid = Gridworld(slip=0.1)
    mdp = grid.model()
    for planning_method in ("value_iteration", "policy_iteration"):
        result = plan(mdp, method=planning_method)
        for iteration, residual in enumerate(result.residuals, 1):
            evidence.add(
                "planning",
                method=planning_method,
                iteration=float(iteration),
                residual=float(residual),
                error_bound=float(residual / 0.05),
            )
        if planning_method == "value_iteration":
            for state, (row, col) in enumerate(grid.cells):
                evidence.add(
                    "grid_values",
                    row=float(row),
                    column=float(col),
                    value=float(result.values[state]),
                    action=float(result.policy[state]),
                )


def control_configs(config: ReinforcementBenchmarkConfig) -> tuple[tuple[str, ControlConfig], ...]:
    """Freeze all candidate settings before any test episode is opened."""

    candidates: list[tuple[str, ControlConfig]] = []
    for repetition in range(config.repetitions):
        seed = (config.seed + repetition) % 2**32
        specifications: tuple[tuple[Literal["reinforce", "dqn", "ppo"], int, float], ...] = (
            ("reinforce", config.reinforce_episodes, 0.005),
            ("dqn", config.dqn_episodes, 0.001),
            ("ppo", config.ppo_episodes, 0.001),
        )
        for method, episodes, rate in specifications:
            control = ControlConfig(
                method=method,
                seed=seed,
                episodes=episodes,
                max_steps=config.control_max_steps,
                learning_rate=rate,
                validation_episodes=config.validation_episodes,
            )
            candidates.append((method, control))
            if method == "dqn" and config.dqn_ablations:
                candidates.extend(
                    [
                        ("dqn_recent", control.model_copy(update={"replay_enabled": False})),
                        ("dqn_online_target", control.model_copy(update={"target_enabled": False})),
                        (
                            "dqn_recent_online",
                            control.model_copy(
                                update={"replay_enabled": False, "target_enabled": False}
                            ),
                        ),
                    ]
                )
    return tuple(candidates)


def _training_evidence(
    name: str, config: ControlConfig, report: ControlReport, evidence: LaboratoryEvidence
) -> None:
    for episode, reward in _window_rows(np.asarray(report.returns)):
        evidence.add(
            "learning",
            method=name,
            seed=float(config.seed),
            episode=float(episode),
            mean_return=reward,
        )
    for record in report.validation:
        evidence.add("validation", method=name, seed=float(config.seed), **record)
    for record in report.diagnostics:
        evidence.add("neural_diagnostics", method=name, seed=float(config.seed), **record)
    evidence.add(
        "resources",
        method=name,
        seed=float(config.seed),
        environment_steps=float(report.environment_steps),
        maximum_interaction_budget=float(config.episodes * config.max_steps),
        selected_episode=float(report.selected_episode),
    )


def _neural(
    config: ReinforcementBenchmarkConfig, evidence: LaboratoryEvidence
) -> tuple[CandidateResult, ...]:
    fitted: list[tuple[str, ControlConfig, ControlReport]] = []
    for name, control in control_configs(config):
        report = train_dqn(control) if control.method == "dqn" else train_policy_gradient(control)
        _training_evidence(name, control, report, evidence)
        fitted.append((name, control, report))
    # All training and within-candidate validation selection are complete before
    # this loop. Test results cannot influence any network or checkpoint choice.
    by_method: dict[str, list[tuple[float, float]]] = {}
    for name, control, report in fitted:
        returns = evaluate_control(
            report.model,
            seed=derive_seed(control.seed, stream=900),
            episodes=config.test_episodes,
            max_steps=config.control_max_steps,
        )
        validation = max(record["mean_return"] for record in report.validation)
        by_method.setdefault(name, []).append((validation, float(returns.mean())))
        for episode, reward in enumerate(returns, 1):
            evidence.add(
                "test_returns",
                method=name,
                seed=float(control.seed),
                episode=float(episode),
                episode_return=float(reward),
            )
        if control.method == "reinforce":
            evidence.add(
                "gradient_variance",
                method=name,
                seed=float(control.seed),
                **baseline_variance(report.model, control, episodes=config.variance_episodes),
            )
    for repetition in range(config.repetitions):
        seed = (config.seed + repetition) % 2**32
        returns = evaluate_control(
            None,
            seed=derive_seed(seed, stream=900),
            episodes=config.test_episodes,
            max_steps=config.control_max_steps,
        )
        by_method.setdefault("random", []).append((0.0, float(returns.mean())))
        for episode, reward in enumerate(returns, 1):
            evidence.add(
                "test_returns",
                method="random",
                seed=float(seed),
                episode=float(episode),
                episode_return=float(reward),
            )
    candidates = []
    for name, pairs in by_method.items():
        values = np.array(pairs)
        rng = np.random.default_rng(derive_seed(config.seed, stream=901))
        bootstrap = rng.choice(values[:, 1], size=(2000, len(values)), replace=True).mean(axis=1)
        metrics = {
            "validation_return": float(values[:, 0].mean()),
            "test_return": float(values[:, 1].mean()),
            "test_seed_sd": float(values[:, 1].std(ddof=1)) if len(values) > 1 else 0.0,
            "test_bootstrap95_low": float(np.quantile(bootstrap, 0.025)),
            "test_bootstrap95_high": float(np.quantile(bootstrap, 0.975)),
            "training_seeds": float(len(values)),
            "seeds_above_cartpole_475": float(np.sum(values[:, 1] >= 475))
            if config.control_max_steps == 500
            else 0.0,
        }
        candidates.append(CandidateResult(name=name, metrics=metrics))
        evidence.add("summary", method=name, **metrics)
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class ReinforcementBenchmark:
    """Paradigm-native environment experiment using the existing atomic runner."""

    config: ReinforcementBenchmarkConfig

    def run(self, context: RunContext) -> RunResult:
        if context.seed != self.config.seed:
            raise ValueError("run context seed must match reinforcement configuration")
        evidence = LaboratoryEvidence()
        _classical(self.config, evidence)
        candidates = _neural(self.config, evidence)
        # Ablations diagnose DQN; they are not extra candidates in a winner search.
        selected = max(
            (
                candidate
                for candidate in candidates
                if candidate.name in {"reinforce", "dqn", "ppo"}
            ),
            key=lambda candidate: (candidate.metrics["validation_return"], candidate.name),
        )
        settings = self.config.model_dump(mode="json")
        context.artifacts.write_json(
            "evidence.json",
            {
                "schema_version": "learning-atlas-rl-evidence/1",
                "config": settings,
                "tables": evidence.tables,
            },
        )
        comparison = io.StringIO(newline="")
        writer = csv.DictWriter(comparison, fieldnames=["method", *sorted(candidates[0].metrics)])
        writer.writeheader()
        writer.writerows({"method": item.name, **item.metrics} for item in candidates)
        context.artifacts.write_text("comparison.csv", comparison.getvalue())
        lines = [
            "# Reinforcement-learning comparison",
            "",
            "Training-validation selects policies; untouched test never controls selection.",
            "",
            "| Method | Validation return | Test return | Training-seed SD | Seeds >=475 |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in candidates:
            metrics = item.metrics
            lines.append(
                f"| {item.name} | {metrics['validation_return']:.2f} | {metrics['test_return']:.2f} | {metrics['test_seed_sd']:.2f} | {metrics['seeds_above_cartpole_475']:.0f} |"
            )
        lines.extend(
            [
                "",
                "The 475 threshold applies only to the full 500-step CartPole profile.",
                "Seed-level bootstrap intervals are descriptive and fragile with few policies.",
                "See evidence.json for every seed and measured interaction cost.",
                "",
            ]
        )
        context.artifacts.write_text("comparison.md", "\n".join(lines))
        plots = publish_reinforcement_plots(evidence.tables, context.artifacts)
        return RunResult(
            experiment=self.config.experiment,
            paradigm=LearningParadigm.REINFORCEMENT,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.ENVIRONMENT,
                name="Atlas stationary bandits/Gridworld/CartPole-v1",
                version="1",
                fingerprint_sha256=hashlib.sha256(
                    json.dumps(settings, sort_keys=True).encode()
                ).hexdigest(),
                target_used_for_fit=None,
                details={
                    "offline": True,
                    "control_max_steps": self.config.control_max_steps,
                    "selection_stream": 20,
                    "test_stream": 900,
                    "variance_stream": 400,
                },
            ),
            selected_model=selected.name,
            metrics=selected.metrics,
            candidates=candidates,
            artifacts={
                "evidence": "evidence.json",
                "comparison_csv": "comparison.csv",
                "comparison_markdown": "comparison.md",
                **plots,
            },
            notes=(
                "Synthetic environments only; no trading, robotics, or production readiness inference.",
                "Training-validation selects checkpoints and the headline method; test is never used for selection.",
                "Uncertainty resamples training-seed means, not falsely independent episodes; few-seed intervals are fragile.",
                "Ablations share maximum episode/interaction and optimizer settings; realized interactions differ with survival.",
                "Reduced CI budgets validate mechanics, not the CartPole 475/500 solve criterion.",
            ),
        )
