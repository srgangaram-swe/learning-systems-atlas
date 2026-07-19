"""Command-line interface for discovering and running atlas experiments."""

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from learning_atlas import __version__
from learning_atlas.core.config import config_schema, load_config
from learning_atlas.core.registry import REGISTRY
from learning_atlas.core.runner import run_experiment
from learning_atlas.supervised.comparison import CandidateExecutionError
from learning_atlas.unsupervised.comparison import UnsupervisedCandidateError
from learning_atlas.workflows.suite import run_suite
from learning_atlas.workflows.supervised_benchmark import run_supervised_benchmark
from learning_atlas.workflows.unsupervised_benchmark import run_unsupervised_benchmark

app = typer.Typer(
    name="learning-atlas",
    help="Run reproducible supervised, unsupervised, and reinforcement-learning benchmarks.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def application(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Learning Systems Atlas command group."""


@app.command("list")
def list_experiments() -> None:
    """List registered reference experiments."""

    for spec in REGISTRY.values():
        typer.echo(f"{spec.name}\t{spec.paradigm.value}\t{spec.description}")


@app.command()
def validate(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a YAML config and print its normalized representation."""

    try:
        config = load_config(config_path)
    except (
        OSError,
        ValueError,
        ValidationError,
        CandidateExecutionError,
        UnsupervisedCandidateError,
    ) as error:
        typer.echo(f"invalid configuration: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def schema() -> None:
    """Print the current experiment-configuration JSON schema."""

    typer.echo(json.dumps(config_schema(), indent=2, sort_keys=True))


@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
) -> None:
    """Execute one validated experiment into a new artifact directory."""

    try:
        result = run_experiment(load_config(config_path), output_dir)
    except (
        OSError,
        ValueError,
        ValidationError,
        CandidateExecutionError,
        UnsupervisedCandidateError,
    ) as error:
        typer.echo(f"run failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("run-all")
def run_all(
    config_dir: Annotated[Path, typer.Option("--config-dir", "-c")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
) -> None:
    """Run every YAML config under a directory and print a compact suite summary."""

    try:
        results = run_suite(config_dir, output_dir)
    except (
        OSError,
        ValueError,
        ValidationError,
        CandidateExecutionError,
        UnsupervisedCandidateError,
    ) as error:
        typer.echo(f"suite failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    summary = {
        result.experiment: {
            "paradigm": result.paradigm.value,
            "selected_model": result.selected_model,
            "metrics": result.metrics,
        }
        for result in results
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("benchmark-supervised")
def benchmark_supervised(
    config_dir: Annotated[Path, typer.Option("--config-dir", "-c")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
) -> None:
    """Run and atomically publish both Sprint 2 from-scratch comparisons."""

    try:
        results = run_supervised_benchmark(config_dir, output_dir)
    except (OSError, ValueError, ValidationError, CandidateExecutionError) as error:
        typer.echo(f"supervised benchmark failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    summary = {
        result.experiment: {
            "selected_model": result.selected_model,
            "metrics": result.metrics,
        }
        for result in results
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("benchmark-unsupervised")
def benchmark_unsupervised(
    config_dir: Annotated[Path, typer.Option("--config-dir", "-c")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
) -> None:
    """Run and atomically publish both Sprint 3 unsupervised studies."""

    try:
        results = run_unsupervised_benchmark(config_dir, output_dir)
    except (
        OSError,
        ValueError,
        ValidationError,
        CandidateExecutionError,
        UnsupervisedCandidateError,
    ) as error:
        typer.echo(f"unsupervised benchmark failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    summary = {
        result.experiment: {
            "selected_model": result.selected_model,
            "metrics": result.metrics,
        }
        for result in results
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Console-script entry point."""

    app()
