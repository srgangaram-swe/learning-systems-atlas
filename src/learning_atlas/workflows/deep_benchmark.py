"""Atomic publication for the Sprint 4 deep-learning evidence suite."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from learning_atlas.core.contracts import RunResult
from learning_atlas.core.reproducibility import environment_metadata
from learning_atlas.workflows.suite import run_suite

_EXPECTED_EXPERIMENTS = {
    "scratch_mlp_benchmark",
    "deep_vision_benchmark",
    "deep_sequence_benchmark",
    "deep_autoencoder_benchmark",
}


def _validate_results(results: tuple[RunResult, ...]) -> None:
    observed = {result.experiment for result in results}
    if observed != _EXPECTED_EXPERIMENTS or len(results) != len(_EXPECTED_EXPERIMENTS):
        expected = ", ".join(sorted(_EXPECTED_EXPERIMENTS))
        actual = ", ".join(sorted(observed)) or "none"
        msg = f"deep benchmark requires exactly {expected}; received {actual}"
        raise ValueError(msg)


def _write_reports(root: Path, results: tuple[RunResult, ...]) -> None:
    payload = {
        result.experiment: {
            "selected_model": result.selected_model,
            "metrics": result.metrics,
            "candidates": [candidate.model_dump(mode="json") for candidate in result.candidates],
            "source": result.source.model_dump(mode="json"),
            "notes": list(result.notes),
        }
        for result in results
    }
    (root / "comparison.json").write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, str | float]] = []
    for result in results:
        for candidate in result.candidates:
            row: dict[str, str | float] = {
                "experiment": result.experiment,
                "candidate": candidate.name,
                "selected": str(candidate.name == result.selected_model).lower(),
            }
            row.update(candidate.metrics)
            rows.append(row)
    metric_columns = sorted(
        {key for row in rows for key in row if key not in {"experiment", "candidate", "selected"}}
    )
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["experiment", "candidate", "selected", *metric_columns],
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    by_name = {result.experiment: result for result in results}
    rows_md = (
        (
            "From-scratch MLP",
            by_name["scratch_mlp_benchmark"],
            "mean_validation_accuracy",
            "mean_test_accuracy",
        ),
        (
            "Image classification",
            by_name["deep_vision_benchmark"],
            "selection_validation_accuracy",
            "test_accuracy",
        ),
        (
            "Sequence classification",
            by_name["deep_sequence_benchmark"],
            "selection_validation_accuracy",
            "test_accuracy",
        ),
        (
            "Representation/anomaly",
            by_name["deep_autoencoder_benchmark"],
            "selection_validation_auroc",
            "anomaly_auroc",
        ),
    )
    lines = [
        "# Sprint 4 deep-learning benchmark",
        "",
        "Candidate selection uses training-only validation evidence. Untouched test metrics",
        "are opened once after selection and are never used for tuning or winner choice.",
        "",
        "| Study | Selected model | Validation selection evidence | Untouched-test evidence |",
        "|---|---|---:|---:|",
    ]
    for study, result, validation_metric, test_metric in rows_md:
        lines.append(
            f"| {study} | {result.selected_model} | "
            f"{validation_metric} {result.metrics[validation_metric]:.4f} | "
            f"{test_metric} {result.metrics[test_metric]:.4f} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(root: Path, results: tuple[RunResult, ...]) -> None:
    manifest_path = root / "benchmark_manifest.json"
    checksums = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "workflow": "sprint-04-deep-learning-benchmark",
        "experiments": [
            {"name": result.experiment, "seed": result.seed}
            for result in sorted(results, key=lambda item: item.experiment)
        ],
        "environment": environment_metadata(),
        "root_reports": ["comparison.csv", "comparison.json", "comparison.md"],
        "artifacts_sha256": checksums,
    }
    manifest_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_deep_benchmark(config_dir: Path, output_dir: Path) -> tuple[RunResult, ...]:
    """Run all Sprint 4 studies and atomically publish aggregate evidence."""

    target = output_dir.resolve()
    if target.exists() and any(target.iterdir()):
        msg = f"refusing to overwrite non-empty benchmark directory: {target}"
        raise FileExistsError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        results = run_suite(config_dir, staging / "experiments")
        _validate_results(results)
        _write_reports(staging, results)
        _write_manifest(staging, results)
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return results
